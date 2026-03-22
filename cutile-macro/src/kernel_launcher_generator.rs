/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Kernel launcher generation.
//!
//! This module generates launcher functions for GPU kernel entry points
//! with support for both synchronous and asynchronous execution.
//! These launchers provide a type-safe interface for invoking
//! CUDA Tile kernels from Rust code.
//!
//! ## Overview
//!
//! For each function marked with `#[cutile::entry]`, this module generates
//! an `_apply` function that:
//!
//! 1. **Compiles** the kernel (with caching)
//! 2. **Infers** generic parameters from input types
//! 3. **Handles** tensor partitioning and grid inference
//! 4. **Launches** the kernel asynchronously
//! 5. **Returns** results as device operations
//!
//! ## Generated Launcher Structure
//!
//! ```rust,ignore
//! // For this kernel:
//! #[cutile::entry]
//! fn my_kernel<T: ElementType, const N: i32>(
//!     output: &mut Tensor<T, {[N]}>,
//!     input: &Tensor<T, {[-1]}>,
//! ) { }
//!
//! // Generates this launcher:
//! pub fn my_kernel_apply<T: Send + WithDType>(
//!     inputs: (Partition<Tensor<T>>, Arc<Tensor<T>>),
//! ) -> impl DeviceOperation<Output=(Partition<Tensor<T>>, Arc<Tensor<T>>)> + TileKernel<...> {
//!     // Compilation and launch logic
//! }
//! ```
//!
//! ## Generic Parameter Inference
//!
//! The launcher automatically infers generic parameters from input tensors:
//!
//! - **Type parameters** (`T`) - Inferred from tensor element types
//! - **Const scalars** (`const N: i32`) - Inferred from tensor shapes
//! - **Const arrays** (`const S: [i32; N]`) - Inferred from partition shapes
//!
//! ## Parameter Transformation
//!
//! Kernel parameters are transformed for the launcher:
//!
//! - `&mut Tensor<T, S>` → `Partition<Tensor<T>>` (mutable, partitioned)
//! - `&Tensor<T, S>` → `Arc<Tensor<T>>` (immutable, shared)
//! - Scalars remain unchanged
//!
//! ## Grid Inference
//!
//! Launch grid dimensions are automatically inferred from partitioned tensors.
//! If multiple partitions exist, their grids must match.

use cutile_compiler::syn_utils::*;
use cutile_compiler::types::get_ptr_type;
use proc_macro2::Ident;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use std::collections::HashMap;
use syn::{
    parse_quote, AngleBracketedGenericArguments, Expr, ExprBlock, FnArg, GenericArgument,
    GenericParam, Generics, ImplItemFn, ItemFn, PatType, Stmt, Type, TypeParam, TypeReference,
    WherePredicate,
};

use crate::error::{Error, SpannedError};

/// Classification of generic parameter types.
///
/// Used to determine how to handle and infer different kinds of generic parameters.
#[derive(Debug, PartialOrd, PartialEq)]
pub(crate) enum SupportedGenericType {
    /// Type parameter (e.g., `T: ElementType`)
    TypeParam,
    /// Const scalar parameter (e.g., `const N: i32`)
    ConstScalar,
    /// Const array parameter (e.g., `const S: [i32; 2]`)
    ConstArray,
    /// Unknown or unsupported parameter type
    Unknown,
}

/// Tracks required generic parameters and their inference state.
///
/// Manages the set of generic parameters that need to be inferred from
/// kernel arguments. Tracks which parameters have been successfully inferred
/// and generates the final generic argument list.
#[derive(Debug)]
pub(crate) struct RequiredGenerics {
    /// Names of generic parameters in order
    names: Vec<String>,
    /// Names of compile-time generic type parameters required by launcher functions.
    /// This lets generated launcher functions support type param inference via impls of the WithDType trait.
    launcher_type_params: Vec<String>,
    /// Type annotations for const generic parameters (None for type params)
    types: Vec<Option<Type>>,
    /// Inferred expressions for each generic parameter
    expressions: HashMap<String, Option<String>>,
}

impl RequiredGenerics {
    fn new(generics: &Generics) -> Self {
        let req_generics = get_supported_generic_params(generics);
        let (names, types): (Vec<String>, _) = req_generics.into_iter().unzip();
        let mut expressions: HashMap<String, Option<String>> = HashMap::new();
        for name in &names {
            expressions.insert(name.clone(), None);
        }
        Self {
            names,
            launcher_type_params: vec![],
            types,
            expressions,
        }
    }
    /// Returns `true` if the given generic parameter has not yet been inferred.
    pub(crate) fn is_required(&self, s: &str) -> bool {
        matches!(self.expressions.get(s), None | Some(None))
    }
    /// Builds a runtime expression string that concatenates all inferred generic values.
    pub(crate) fn to_expr_str(&self) -> String {
        let mut res = vec![];
        for name in &self.names {
            let Some(Some(expr_str)) = self.expressions.get(name) else {
                // TODO (hme): Some generics not assigned in function signature can't be inferred.
                return format!("panic!(\"Failed to infer value for generic parameter {name}\")");
            };
            res.push(expr_str.clone());
        }
        format!("vec![{}].concat()", res.join(","))
    }

    /// Classifies the generic parameter with the given name.
    pub(crate) fn get_ty(&self, name: &str) -> SupportedGenericType {
        let Some(index) = self.names.iter().position(|n| n == name) else {
            return SupportedGenericType::Unknown;
        };
        let Some(ty) = &self.types[index] else {
            // If there is no type, it's not a const VAR_NAME: Type.
            // It is a type param.
            return SupportedGenericType::TypeParam;
        };
        match ty {
            Type::Array(_) => SupportedGenericType::ConstArray,
            _ => SupportedGenericType::ConstScalar,
        }
    }
    /// Builds a `Generics` containing only the type parameters needed by the launcher.
    pub(crate) fn get_required_generics(&self) -> Generics {
        let mut type_params = vec![];
        for name in &self.names {
            let is_launcher_type_param = self.launcher_type_params.contains(name);
            if is_launcher_type_param && self.get_ty(name) == SupportedGenericType::TypeParam {
                type_params.push(format!("{}: Send + WithDType", name.clone()));
            }
        }
        syn::parse2::<Generics>(format!("<{}>", type_params.join(", ")).parse().unwrap()).unwrap()
    }
    /// Builds angle-bracketed generic arguments for the launcher type parameters.
    pub(crate) fn get_generic_args(&self) -> AngleBracketedGenericArguments {
        let mut type_params = vec![];
        for name in &self.names {
            let is_launcher_type_param = self.launcher_type_params.contains(name);
            if is_launcher_type_param && self.get_ty(name) == SupportedGenericType::TypeParam {
                type_params.push(name.to_string());
            }
        }
        syn::parse2::<AngleBracketedGenericArguments>(
            format!("<{}>", type_params.join(", ")).parse().unwrap(),
        )
        .unwrap()
    }
}

/// Joins a vector of strings into a nested cons-cell tuple structure.
///
/// Converts a list of values into a right-associated nested tuple structure
/// used by async combinators. For example, `["a", "b", "c"]` becomes `"(a, (b, c))"`.
///
/// ## Parameters
///
/// - `vals`: Vector of string representations of values to join
///
/// ## Returns
///
/// A string representing the nested tuple structure
///
/// ## Examples
///
/// - `[]` → `"()"`
/// - `["a"]` → `"a"`
/// - `["a", "b"]` → `"(a, b)"`
/// - `["a", "b", "c"]` → `"(a, (b, c))"`
pub fn join_as_cons_tuple(vals: &[String]) -> String {
    if vals.is_empty() {
        return "()".to_string();
    }
    if vals.len() == 1 {
        return vals[0].clone();
    };
    let mut cons = vals.last().expect("Impossible").clone();
    for i in (0..vals.len() - 1).rev() {
        cons = format!("({}, {})", vals[i], cons);
    }
    cons
}

/// Wraps an expression in a `value()` call if needed for async combinators.
///
/// Helper function used when building async launcher code. If `wrap_as_val` is true,
/// wraps the expression in `value()` to create an async value.
///
/// ## Parameters
///
/// - `expr`: The expression string to potentially wrap
/// - `wrap_as_val`: Whether to wrap the expression in `value()`
///
/// ## Returns
///
/// Either the original expression or `value(expr)`
fn zippable(expr: &str, wrap_as_val: bool) -> String {
    if !wrap_as_val {
        return expr.to_string();
    }
    format!("value({})", expr)
}

/// Generates async code to zip inputs into a cons-cell tuple structure.
///
/// Creates a block of statements that uses the `zip!` macro to combine async
/// values into a nested tuple. Each input is zipped with the accumulator in
/// reverse order to create a right-associated structure.
///
/// ## Parameters
///
/// - `inputs`: Vector of input expression strings to zip
/// - `var_name`: Name of the variable to bind the zipped result to
/// - `wrap_as_val`: Whether to wrap inputs in `value()` calls
///
/// ## Returns
///
/// An expression block containing the zip statements
///
/// ## Note
///
/// Currently unused (marked with `#[allow(dead_code)]`). See `zip_and_then_flatten`
/// for the actively used version.
#[allow(dead_code)]
pub fn zip_cons(inputs: &[String], var_name: &str, wrap_as_val: bool) -> ExprBlock {
    let mut zip_block = syn::parse2::<ExprBlock>(quote! {{
    }})
    .unwrap();
    if inputs.is_empty() {
        return zip_block;
    }
    let mut i = inputs.len() - 1;
    zip_block.block.stmts.push(parse_stmt(format!(
        "let {var_name} = {};",
        zippable(&inputs[i], wrap_as_val)
    )));
    while i != 0 {
        i -= 1;
        zip_block.block.stmts.push(parse_stmt(format!(
            "let {var_name} = zip!({}, {var_name});",
            zippable(&inputs[i], wrap_as_val)
        )));
    }
    zip_block
}

/// Generates async code to zip inputs, then flatten them into a flat tuple.
///
/// Similar to `zip_cons` but adds a final `and_then` step that flattens the
/// nested cons-cell structure into a regular flat tuple. This is the standard
/// approach used for async kernel launchers.
///
/// ## Parameters
///
/// - `inputs`: Vector of input expression strings to zip
/// - `var_name`: Name of the variable to bind the result to
/// - `wrap_as_val`: Whether to wrap inputs in `value()` calls
///
/// ## Returns
///
/// An expression block containing:
/// 1. Zip statements building a nested tuple
/// 2. An `and_then` that flattens it to a flat tuple
///
/// ## Example
///
/// For inputs `["a", "b", "c"]`, generates code equivalent to:
/// ```rust,ignore
/// let result = zip!(a, zip!(b, c));
/// let result = result.and_then(|(a, (b, c))| value((a, b, c)));
/// ```
pub fn zip_and_then_flatten(inputs: &[String], var_name: &str, wrap_as_val: bool) -> ExprBlock {
    let mut zip_block = syn::parse2::<ExprBlock>(quote! {{
    }})
    .unwrap();
    if inputs.is_empty() {
        zip_block
            .block
            .stmts
            .push(parse_stmt(format!("let {var_name} = value(());")));
        return zip_block;
    }
    let mut i = inputs.len() - 1;
    zip_block.block.stmts.push(parse_stmt(format!(
        "let {var_name} = {};",
        zippable(&inputs[i], wrap_as_val)
    )));
    while i != 0 {
        i -= 1;
        zip_block.block.stmts.push(parse_stmt(format!(
            "let {var_name} = zip!({}, {var_name});",
            zippable(&inputs[i], wrap_as_val)
        )));
    }
    zip_block.block.stmts.push(parse_stmt(format!(
        r#"
            let {var_name} = {var_name}.and_then(|{}| {{
                value({})
            }});
        "#,
        join_as_cons_tuple(inputs),
        to_tuple_string(inputs)
    )));
    zip_block
}

/// Generates a type alias name for a kernel argument.
///
/// Helper function for creating unique type alias names for kernel launcher arguments.
/// For example, generates `"FooKernelArg0"`, `"FooKernelArg1"`, etc.
///
/// ## Parameters
///
/// - `launcher_name`: The name of the kernel launcher
/// - `i`: The argument index
///
/// ## Returns
///
/// A string like `"LauncherNameArgN"`
///
/// ## Note
///
/// Currently unused (marked with `#[allow(dead_code)]`).
#[allow(dead_code)]
fn kernel_arg_alias(launcher_name: &str, i: usize) -> String {
    format!("{launcher_name}Arg{i}")
}

/// Generates a type alias for kernel launcher arguments.
///
/// Creates a type alias that represents the tuple of all kernel arguments,
/// including any generic parameters from the kernel function signature.
///
/// ## Parameters
///
/// - `generic_args`: Generic parameters from the kernel (e.g., `<T, const N: i32>`)
/// - `arg_tys`: Types of all kernel arguments
/// - `_launcher_name`: Name of the launcher (currently unused)
/// - `launcher_args_name`: Name for the type alias (e.g., `"FooKernelArgs"`)
///
/// ## Returns
///
/// A tuple of:
/// 1. The constructed type (e.g., `FooKernelArgs<T, N>`)
/// 2. The type alias definition as tokens (e.g., `type FooKernelArgs<T, N> = (Tensor<T, [N]>, i32);`)
pub fn generate_launcher_arg_types(
    generic_args: &AngleBracketedGenericArguments,
    arg_tys: &Vec<Type>,
    _launcher_name: &str,
    launcher_args_name: &str,
) -> (Type, TokenStream2) {
    let launcher_args_ident = Ident::new(launcher_args_name, Span::call_site());
    let launcher_args_type: Type = if !generic_args.args.is_empty() {
        parse_quote! { #launcher_args_ident #generic_args }
    } else {
        parse_quote! { #launcher_args_ident }
    };
    (
        launcher_args_type.clone(),
        quote! { type #launcher_args_type = ( #(#arg_tys,)* ); },
    )
}

/// Converts a vector of strings into a flat tuple string representation.
///
/// Helper function that formats argument names into a comma-separated tuple string.
/// Used when generating async launcher code.
///
/// ## Parameters
///
/// - `args`: Vector of argument names/expressions
///
/// ## Returns
///
/// A string like `"(arg1, arg2, arg3,)"` with trailing comma
///
/// ## Example
///
/// `["x", "y", "z"]` → `"(x, y, z,)"`
pub fn to_tuple_string(args: &[String]) -> String {
    format!(
        "({})",
        args.iter()
            .map(|s| format!("{s},"))
            .collect::<Vec<String>>()
            .join("")
    )
}

/// Generates the complete async launcher code for a GPU kernel.
///
/// This is the main function that transforms a kernel function (marked with `#[entry]`)
/// into an async launcher that can be called from Rust code. It generates:
/// 1. Type aliases for kernel arguments
/// 2. A launcher struct implementing `DeviceOperation`
/// 3. An `execute` method that builds the kernel call and launches it
///
/// ## Parameters
///
/// - `item`: The kernel function AST
/// - `module_name`: Name of the module containing the kernel
/// - `function_name`: Name of the kernel function
/// - `function_entry_name`: MLIR entry point name for the kernel
/// - `launcher_name`: Name for the generated launcher struct
/// - `launcher_args_name`: Name for the generated argument type alias
///
/// ## Returns
///
/// A tuple of:
/// 1. `RequiredGenerics` - Information about generic parameters and constraints
/// 2. `(Type, TokenStream2)` - The launcher argument type and its definition
/// 3. `TokenStream2` - The complete launcher implementation
///
/// ## Generated Code Structure
///
/// For a kernel `fn my_kernel<T>(x: &mut Tensor<T, [128]>, y: &Tensor<T, [-1]>)`, generates:
/// ```rust,ignore
/// type MyKernelArgs<T> = (Arc<Tensor<T>>, Arc<Tensor<T>>);
/// pub struct MyKernel<T> {
///     args: MyKernelArgs<T>,
/// }
/// impl<T: WithDType> DeviceOperation for MyKernel<T> {
///     type Output = ();
///     unsafe fn execute(mut self, ctx: &ExecutionContext) -> Self::Output {
///         // Async launcher logic here
///     }
/// }
/// ```
pub fn generate_kernel_launcher(
    item: &ItemFn,
    module_name: &str,
    function_name: &str,
    function_entry_name: &str,
    launcher_name: &str,
    launcher_args_name: &str,
) -> Result<(RequiredGenerics, (Type, TokenStream2), TokenStream2), Error> {
    let unsafety = item.sig.unsafety;
    let is_unsafe = unsafety.is_some();
    let launcher_ident = Ident::new(launcher_name, Span::call_site());
    let mut launcher_method = syn::parse2::<ImplItemFn>(quote! {
        unsafe fn execute(mut self, ctx: &ExecutionContext) -> Result<<Self as DeviceOperation>::Output, DeviceError> {}
    })
    .unwrap();

    // Generate launcher signature.
    let param_names = get_sig_param_names(&item.sig);
    let param_names_tuple_str = to_tuple_string(&param_names);
    let (input_types, _output_type) = get_sig_types(&item.sig, None);
    let mut stride_args = vec![];
    let mut builder_statements = vec![];
    let mut launch_grid_expr_strs = vec![];
    let mut validator_statements = vec![];
    let mut arg_types: Vec<Type> = vec![];

    let mut required_generics: RequiredGenerics = RequiredGenerics::new(&item.sig.generics);
    // println!("required_generics: {}", required_generics.get_required_generics().to_token_stream().to_string());
    for (i, (var_name, ty)) in param_names.iter().zip(input_types.iter()).enumerate() {
        // Currently only supporting scalars, &Tensor, and &mut Tensor.
        // This should be enough to do everything safely.
        // Added support for * mut T to allow for unsafe kernels.
        match ty {
            Type::Reference(ref_ty) => {
                let res = get_tensor_code(i, var_name, ref_ty, &mut required_generics)?;
                arg_types.push(res.fn_arg.ty.as_ref().clone());
                stride_args.push(res.stride_expr_str);
                builder_statements.extend(res.builder_statements);
                launch_grid_expr_strs.extend(res.launch_grid_expr_strs);
                validator_statements.extend(res.validator_statements.block.stmts);
            }
            Type::Path(path_ty) => {
                let ident = get_ident_from_path(&path_ty.path);
                let type_name = ident.to_string();
                arg_types.push(syn::parse2::<Type>(type_name.parse().unwrap()).unwrap());
                if required_generics.is_required(&type_name) {
                    required_generics
                        .launcher_type_params
                        .push(type_name.clone());
                    // T is WithDType, so we can use T::DTYPE.as_str();
                    required_generics.expressions.insert(
                        type_name.clone(),
                        Some(format!("vec![{type_name}::DTYPE.as_str().to_string()]")),
                    );
                }
                builder_statements.push(parse_stmt(format!(
                    "kernel_launch.push_arg(Box::new({var_name}));"
                )));
            }
            Type::Ptr(ptr_type) => {
                // Let's require this to be unsafe, even though all unsafe operations on pointers
                // are marked as such.
                if !is_unsafe {
                    return ptr_type
                        .err("Pointers can only be used in unsafe kernel entry points.");
                }
                let ptr_str = ptr_type.to_token_stream().to_string();
                let Some((is_mutable, type_name)) = get_ptr_type(&ptr_str) else {
                    return ptr_type.err(&format!("Unexpected pointer type: {}", ptr_str));
                };
                if !is_mutable {
                    return ptr_type.err("Pointers must be * mut.");
                }
                arg_types.push(
                    syn::parse2::<Type>(format!("DevicePointer<{}>", type_name).parse().unwrap())
                        .unwrap(),
                );
                if required_generics.is_required(&type_name) {
                    required_generics
                        .launcher_type_params
                        .push(type_name.clone());
                    // T is WithDType, so we can use T::DTYPE.as_str();
                    required_generics.expressions.insert(
                        type_name.clone(),
                        Some(format!("vec![{type_name}::DTYPE.as_str().to_string()]")),
                    );
                }
                builder_statements.push(parse_stmt(format!("kernel_launch.push_arg({var_name});")));
            }
            _ => {
                return ty.err("Unable to generate launcher: unsupported parameter type.");
            }
        }
    }

    // Prepare generics.
    let generic_params = required_generics.get_required_generics();
    let generic_args = required_generics.get_generic_args();
    let (launcher_args_type, launcher_arg_type_def) =
        generate_launcher_arg_types(&generic_args, &arg_types, launcher_name, launcher_args_name);
    let launcher_args_type_str = launcher_args_type.to_token_stream().to_string();
    let device_op_param: GenericParam =
        parse_quote! { DI: DeviceOperation<Output=#launcher_args_type> };
    let device_op_arg: GenericArgument = parse_quote! { DI };
    let mut struct_generics = generic_params.clone();
    struct_generics.params.push(device_op_param.clone());
    let mut struct_args = generic_args.clone();
    struct_args.args.push(device_op_arg.clone());
    let mut launch_output_type = generic_args.clone();
    let impl_device_op: GenericArgument =
        parse_quote! { impl DeviceOperation<Output=#launcher_args_type> };
    launch_output_type.args.push(impl_device_op);

    let init_stmts = syn::parse2::<ExprBlock>(quote! {{
        let module_name = #module_name;
        let function_name = #function_name;
        let function_entry = #function_entry_name;
        let input = self.input.take().unwrap();
    }})
    .unwrap()
    .block
    .stmts;
    launcher_method.block.stmts.extend(init_stmts);
    launcher_method.block.stmts.push(parse_stmt(format!(
        r#"let {param_names_tuple_str}: {launcher_args_type_str} = input.execute(ctx)?;"#
    )));

    if !required_generics.names.is_empty() {
        launcher_method.block.stmts.push(parse_stmt(format!(
            r#"
            let function_generics: Vec<String> = if self.function_generics.is_some() {{
                self.function_generics.take().unwrap()
            }} else {{
                {}
            }};
            "#,
            required_generics.to_expr_str()
        )));
    } else {
        launcher_method.block.stmts.push(parse_stmt(
            "let function_generics: Vec<String> = vec![];".to_string(),
        ));
    }

    launcher_method.block.stmts.push(parse_stmt(format!(
        "let stride_args: Vec<(String, Vec<i32>)> =  vec![{}];",
        stride_args.join(",")
    )));

    let compile_stmts = syn::parse2::<ExprBlock>(quote! {{
        let const_grid = if self._const_grid { Some(self._grid) } else { None };
        let (function, validator) = self.compile(
            ctx, _module_asts,
            module_name, function_name, function_entry,
            function_generics, stride_args, const_grid
        )?;
        // Do validation here.
    }})
    .unwrap()
    .block
    .stmts;
    launcher_method.block.stmts.extend(compile_stmts);
    launcher_method.block.stmts.extend(validator_statements);

    // Add launcher arguments.
    launcher_method.block.stmts.push(parse_stmt(
        "let mut kernel_launch = AsyncKernelLaunch::new(function.clone());".to_string(),
    ));
    launcher_method.block.stmts.extend(builder_statements);

    // Infer launch grid.
    launcher_method.block.stmts.push(parse_stmt(format!(
        "let launch_grid: (u32, u32, u32) = self.infer_launch_grid(&[{}])?;",
        launch_grid_expr_strs.join(",")
    )));

    let launch_stmts = syn::parse2::<ExprBlock>(quote! {{
        // Launch the kernel. This is the same for all functions.
        kernel_launch
            .set_launch_config(LaunchConfig {
                grid_dim: launch_grid,
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0
            });
        kernel_launch.execute(ctx)?;
    }})
    .unwrap()
    .block
    .stmts;
    launcher_method.block.stmts.extend(launch_stmts);
    launcher_method.block.stmts.push(parse_stmt(format!(
        r#"return Ok({param_names_tuple_str});"#
    )));

    // Generate launcher apply function. This is the simplest case.
    let kernel_return_type = quote! {
        #launcher_ident #launch_output_type
    };
    let apply_name = format!("{}_apply", function_name);
    let launcher_apply_ident = Ident::new(
        format!("{}_apply", function_name).as_str(),
        Span::call_site(),
    );
    let launcher_apply = syn::parse2::<ItemFn>(quote! {
        pub #unsafety fn #launcher_apply_ident #struct_generics (input: DI) -> #kernel_return_type {
            return #launcher_ident::launch(input);
        }
    })
    .unwrap();

    // These are the type aliases generated for the argument types for this kernel function.
    let arg_aliases = arg_types
        .iter()
        .map(|i| i.to_token_stream().to_string())
        .collect::<Vec<_>>();

    // Generate launcher async function. Uses apply function.
    // This operates on and returns a flat tuple of arguments.
    let async_name = format!("{}_async", function_name);
    let launcher_async_ident = Ident::new(async_name.as_str(), Span::call_site());
    let mut launcher_async = syn::parse2::<ItemFn>(quote! {
        pub #unsafety fn #launcher_async_ident #generic_params() -> #kernel_return_type {}
    })
    .unwrap();
    let function_params = arg_types
        .iter()
        .enumerate()
        .map(|(i, _)| format!("arg{i}"))
        .collect::<Vec<_>>();
    let type_params = arg_types
        .iter()
        .enumerate()
        .map(|(i, _)| format!("DI{i}"))
        .collect::<Vec<_>>();
    launcher_async.sig.generics.make_where_clause();
    for ((function_param, type_param), arg_alias) in function_params
        .iter()
        .zip(type_params.iter())
        .zip(arg_aliases.iter())
    {
        let type_bound = format!("DeviceOperation<Output={}>", arg_alias);
        launcher_async.sig.inputs.push(FnArg::Typed(
            syn::parse2::<PatType>(
                format!("{}: {}", function_param, type_param)
                    .parse()
                    .unwrap(),
            )
            .unwrap(),
        ));
        launcher_async.sig.generics.params.push(GenericParam::Type(
            syn::parse2::<TypeParam>(type_param.parse().unwrap()).unwrap(),
        ));
        let where_clause = launcher_async
            .sig
            .generics
            .where_clause
            .as_mut()
            .expect("Impossible.");
        where_clause.predicates.push(
            syn::parse2::<WherePredicate>(
                format!("{}: {}", type_param, type_bound).parse().unwrap(),
            )
            .unwrap(),
        );
    }
    let input_zips = zip_and_then_flatten(&function_params, "input", false);
    launcher_async.block.stmts.extend(input_zips.block.stmts);
    launcher_async
        .block
        .stmts
        .push(parse_stmt(format!("return {}(input);", apply_name)));

    // Generate launcher sync function. Uses async function.
    let launcher_sync_ident = Ident::new(
        format!("{}_sync", function_name).as_str(),
        Span::call_site(),
    );
    let mut launcher_sync = syn::parse2::<ItemFn>(quote! {
        pub #unsafety fn #launcher_sync_ident #generic_params() -> #kernel_return_type {}
    })
    .unwrap();
    for (function_param, type_param) in function_params.iter().zip(arg_aliases.iter()) {
        launcher_sync.sig.inputs.push(FnArg::Typed(
            syn::parse2::<PatType>(
                format!("{}: {}", function_param, type_param)
                    .parse()
                    .unwrap(),
            )
            .unwrap(),
        ));
    }
    let return_op = format!(
        "return {async_name}({});",
        function_params
            .iter()
            .map(|var| zippable(var, true))
            .collect::<Vec<String>>()
            .join(", ")
    );

    launcher_sync.block.stmts.push(parse_stmt(return_op));
    Ok((
        required_generics,
        (launcher_args_type.clone(), launcher_arg_type_def),
        quote! {
            impl #struct_generics DeviceOperation for #launcher_ident #struct_args {
                type Output = #launcher_args_type;
                #launcher_method
            }
            #launcher_apply
            #launcher_async
            #launcher_sync
        },
    ))
}

/// Parses a string into a Rust statement AST node.
///
/// Helper function that converts a string representation of a statement
/// into a `syn::Stmt` for insertion into generated code blocks.
///
/// ## Parameters
///
/// - `s`: String containing valid Rust statement code
///
/// ## Returns
///
/// A parsed `Stmt` AST node
fn parse_stmt(s: String) -> Stmt {
    syn::parse::<Stmt>(s.parse().unwrap()).unwrap()
}

/// Parses a string into a Rust expression AST node.
///
/// Helper function that converts a string representation of an expression
/// into a `syn::Expr` for use in generated code.
///
/// ## Parameters
///
/// - `s`: String containing valid Rust expression code
///
/// ## Returns
///
/// A parsed `Expr` AST node
///
/// ## Note
///
/// Currently unused (marked with `#[allow(dead_code)]`).
#[allow(dead_code)]
fn parse_expr(s: String) -> Expr {
    syn::parse::<Expr>(s.parse().unwrap()).unwrap()
}

/// Code generation result for a tensor kernel parameter.
///
/// Contains all the generated code components needed for a single tensor
/// parameter in the async kernel launcher.
///
/// ## Fields
///
/// - `fn_arg`: The launcher function parameter (e.g., `x: Arc<Tensor<T>>`)
/// - `stride_expr_str`: Expression string for computing tensor strides
/// - `builder_statements`: Statements to add to the kernel builder
/// - `launch_grid_expr_strs`: Expressions for computing launch grid dimensions
struct TensorLaunchCode {
    fn_arg: PatType, // FnArg::Typed(PatType)
    stride_expr_str: String,
    builder_statements: Vec<Stmt>,
    launch_grid_expr_strs: Vec<String>,
    validator_statements: ExprBlock,
}

/// Generates launcher code for a tensor kernel parameter.
///
/// Analyzes a tensor parameter and generates all the necessary code for:
/// 1. The launcher function parameter type
/// 2. Stride calculations
/// 3. Kernel builder statements (pushing arguments, metadata)
/// 4. Launch grid dimension calculations
///
/// ## Parameters
///
/// - `var_name`: Name of the parameter variable
/// - `ty`: The tensor reference type (e.g., `&mut Tensor<T, [128]>`)
/// - `required_generics`: Accumulator for tracking required generic constraints
///
/// ## Returns
///
/// A `TensorLaunchCode` struct containing all generated code components
fn get_tensor_code(
    var_idx: usize,
    var_name: &str,
    ty: &TypeReference,
    required_generics: &mut RequiredGenerics,
) -> Result<TensorLaunchCode, Error> {
    // FnArg
    let (type_ident, type_generic_args) = get_ident_generic_args(&Type::Reference(ty.clone()));
    let Some(type_ident) = type_ident else {
        return ty.err("Expected a named type identifier for tensor parameter.");
    };
    if type_ident != "Tensor" {
        return ty.err(&format!("Expected Tensor type, got {}.", type_ident));
    }
    let Some(GenericArgument::Type(syn::Type::Path(element_type_path))) =
        type_generic_args.args.first()
    else {
        return ty.err("Expected generic argument type path for tensor element type.");
    };

    // Infer generics from type data available in this type.
    infer_shape_params_from_tensor_type(
        var_name,
        &type_generic_args,
        required_generics,
        ty.mutability.is_some(),
    )?;

    let dtype = element_type_path
        .path
        .segments
        .last()
        .unwrap()
        .ident
        .to_string();
    let tensor_type = if ty.mutability.is_some() {
        format!("tensor::Partition<tensor::Tensor<{dtype}>>")
    } else {
        format!("Arc<tensor::Tensor<{dtype}>>")
    };
    let fn_arg =
        syn::parse::<PatType>(format!("{var_name}: {tensor_type}").parse().unwrap()).unwrap();
    // Stride expr.
    let stride_expr_str = if ty.mutability.is_some() {
        // TODO (hme): Re-enable this for const stride?
        // format!(r#"("{var_name}".to_string(), {var_name}.partition_strides.to_vec())"#)
        format!(
            r#"(
            "{var_name}".to_string(),
            {{
                let len = {var_name}.partition_strides.len();
                let mut res = vec![-1; len];
                res[len-1] = 1;
                res
            }}
        )"#
        )
    } else {
        // TODO (hme): Re-enable this for const stride?
        // format!(r#"("{var_name}".to_string(), {var_name}.strides.to_vec())"#)
        format!(
            r#"(
        "{var_name}".to_string(),
        {{
            let len = {var_name}.strides.len();
            let mut res = vec![-1; len];
            res[len-1] = 1;
            res
        }}
        )"#
        )
    };

    // Builder and validator statements.
    let var_ident = Ident::new(var_name, Span::call_site());
    let mut builder_statements = vec![];
    let mut launch_grid_expr_strs = vec![];
    let validator_statements = if ty.mutability.is_some() {
        builder_statements.push(parse_stmt(format!("kernel_launch.push_arg(&{var_name});")));
        launch_grid_expr_strs.push(format!("{var_name}.grid()?"));
        syn::parse2::<ExprBlock>(quote! {{
            {
                let ValidParamType::Tensor(tensor_validator) = &validator.params[#var_idx] else {
                    panic!("Unexpected validator type {:#?}", &validator.params[#var_idx]);
                };
                let valid_shape = &tensor_validator.shape;
                let given_shape = &#var_ident.partition_shape;
                kernel_launch_assert(valid_shape.len() == given_shape.len(),
                    format!("{} rank mismatch: Expected {}, got {}", #var_name, valid_shape.len(), given_shape.len()).as_str())?;
                kernel_launch_assert(valid_shape == given_shape,
                    format!("{} partition shape mismatch. Expected {:?}, got {:?}", #var_name, valid_shape, given_shape).as_str())?;
                // TODO (hme): add validation for strides here too.
            }
        }})
        .unwrap()
    } else {
        builder_statements.push(parse_stmt(format!(
            "kernel_launch.push_arg_arc(&{var_name});"
        )));

        syn::parse2::<ExprBlock>(quote! {{
            {
                let ValidParamType::Tensor(tensor_validator) = &validator.params[#var_idx] else {
                    panic!("Unexpected validator type {:#?}", &validator.params[#var_idx]);
                };
                let valid_shape = &tensor_validator.shape;
                let given_shape = &#var_ident.shape;
                kernel_launch_assert(valid_shape.len() == given_shape.len(),
                    format!("{} rank mismatch: Expected {}, got {}", #var_name, valid_shape.len(), given_shape.len()).as_str())?;
                let valid_shape_mixed = zip(valid_shape, given_shape).map(|(&expected, &given)|{
                    if expected == -1 { given } else { expected }
                }).collect::<Vec<_>>();
                let pred = zip(&valid_shape_mixed, given_shape).all(|(&expected, &given)|{
                    expected == given
                });
                kernel_launch_assert(pred,
                    format!("{} partition shape mismatch. Expected {:?}, got {:?}", #var_name, valid_shape_mixed, given_shape).as_str())?;
                // TODO (hme): add validation for strides here too.
            }
        }})
        .unwrap()
    };

    Ok(TensorLaunchCode {
        fn_arg,
        stride_expr_str,
        builder_statements,
        launch_grid_expr_strs,
        validator_statements,
    })
}

/// Infers and registers runtime expressions for tensor shape parameters.
///
/// Analyzes a tensor type's generic arguments and generates runtime expressions
/// to extract shape information from the tensor at runtime. This enables the
/// launcher to provide shape information to the MLIR compiler for kernel
/// instantiation.
///
/// ## Parameters
///
/// - `var_name`: Name of the tensor variable
/// - `type_generic_args`: Generic arguments from the tensor type (e.g., `<T, {[128, N]}>`)
/// - `required_generics`: Accumulator for tracking required generic constraints
/// - `is_mutable`: Whether the tensor parameter is mutable (affects which shape to use)
///
/// ## Behavior
///
/// For each generic argument:
/// - **Type parameters** (e.g., `T`): Generates `var_name.dtype().as_str().to_string()`
/// - **Const array parameters** (e.g., `N`, `S`):
///   - Mutable: Generates `var_name.partition_shape` extraction
///   - Immutable: Generates `var_name.shape` extraction
///
/// ## Example
///
/// For `x: &mut Tensor<T, {[128, N]}>`, generates:
/// - `T` → `x.dtype().as_str().to_string()`
/// - `N` → Extract from `x.partition_shape`
pub fn infer_shape_params_from_tensor_type(
    var_name: &str,
    type_generic_args: &AngleBracketedGenericArguments,
    required_generics: &mut RequiredGenerics,
    is_mutable: bool,
) -> Result<(), Error> {
    // We assume this is a variadic type.
    for generic_arg in &type_generic_args.args {
        match generic_arg {
            GenericArgument::Type(type_param) => {
                // Currently, this is either shape or element_type.
                if let syn::Type::Path(type_path) = type_param {
                    let last_ident = type_path.path.segments.last().unwrap().ident.to_string();
                    match required_generics.get_ty(&last_ident) {
                        SupportedGenericType::TypeParam => {
                            // This is an element type.
                            required_generics
                                .launcher_type_params
                                .push(last_ident.clone());
                            required_generics.expressions.insert(
                                last_ident.clone(),
                                Some(format!("vec![{var_name}.dtype().as_str().to_string()]")),
                            );
                        }
                        SupportedGenericType::ConstArray => {
                            // This is a CGA type.
                            if is_mutable {
                                required_generics.expressions.insert(last_ident.clone(), Some(format!("{var_name}.partition_shape.iter().map(|x| x.to_string()).collect::<Vec<String>>()")));
                            } else {
                                // This might make sense for a small tensor.
                                required_generics.expressions.insert(last_ident.clone(), Some(format!("{var_name}.shape.iter().map(|x| x.to_string()).collect::<Vec<String>>()")));
                            }
                        }
                        SupportedGenericType::ConstScalar => {
                            return type_path.err(
                                "Unexpected constant scalar type in tensor generic argument.",
                            );
                        }
                        SupportedGenericType::Unknown => {}
                    }
                }
            }
            GenericArgument::Const(const_param) => {
                // println!("expand GenericArgument::Const? {const_param:#?}");
                if let Expr::Block(block_expr) = const_param {
                    // This is something like Tensor<E, {[...]}>
                    if block_expr.block.stmts.len() != 1 {
                        return block_expr.err(&format!(
                            "Expected exactly 1 statement in block expression, got {}.",
                            block_expr.block.stmts.len()
                        ));
                    }
                    let statement = &block_expr.block.stmts[0];
                    let Stmt::Expr(statement_expr, _) = statement else {
                        return block_expr
                            .err("Unexpected block expression: expected an expression statement.");
                    };
                    match statement_expr {
                        Expr::Array(array_expr) => {
                            // This is something like Tensor<E, {[1, 2, -1]}>
                            for (i, elem) in array_expr.elems.iter().enumerate() {
                                match elem {
                                    Expr::Lit(_lit) => {
                                        // Nothing to do to build generic arg expressions.
                                        continue;
                                    }
                                    Expr::Unary(_unary_expr) => {
                                        // Nothing to do to build generic arg expressions.
                                        continue;
                                    }
                                    Expr::Path(path) => {
                                        let ident = get_ident_from_path_expr(path).to_string();
                                        match required_generics.get_ty(&ident) {
                                            SupportedGenericType::TypeParam => {
                                                // This is an element type.
                                                return path.err("Unexpected type param in array type expression.");
                                            }
                                            SupportedGenericType::ConstArray => {
                                                // This is a CGA type.
                                                return path.err("Unexpected const generic array param in array type expression.");
                                            }
                                            SupportedGenericType::ConstScalar => {
                                                if is_mutable {
                                                    required_generics.expressions.insert(ident.clone(), Some(format!("vec![{var_name}.partition_shape[{i}].to_string()]")));
                                                } else {
                                                    required_generics.expressions.insert(ident.clone(), Some(format!("vec![{var_name}.shape[{i}].to_string()]")));
                                                }
                                            }
                                            SupportedGenericType::Unknown => {}
                                        }
                                    }
                                    _ => {
                                        return elem.err(
                                            "Unsupported array element in tensor shape expression.",
                                        );
                                    }
                                }
                            }
                        }
                        Expr::Repeat(repeat_expr) => {
                            // TODO (hme): Unclear under what circumstance it would be beneficial to support this.
                            return repeat_expr
                                .err("Repeat expressions in tensor shape are not yet supported.");
                        }
                        _ => {
                            return block_expr.err(
                                "Unexpected block expression in tensor const generic argument.",
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}
