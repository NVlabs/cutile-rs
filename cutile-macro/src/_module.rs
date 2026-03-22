/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Main module macro implementation and orchestration.
//!
//! This module orchestrates the entire macro expansion pipeline,
//! including validation, AST generation, and launcher creation.
//!
//! ## Module Processing
//!
//! The `module` function is the main entry point that:
//!
//! 1. **Parses** the module and its attributes
//! 2. **Validates** syntax and type constraints
//! 3. **Transforms** items (functions, structs, traits, impls)
//! 4. **Generates** MLIR AST builders
//! 5. **Creates** async kernel launchers for entry points
//! 6. **Emits** the expanded code
//!
//! ## Item Processing
//!
//! Different item types are processed differently:
//!
//! - **Functions** - Generate concrete functions, AST builders, and launchers (if `#[entry]`)
//! - **Structs** - Handle variadic expansion and desugaring of const generics
//! - **Traits** - Process variadic traits and type metadata
//! - **Impls** - Expand variadic implementations
//! - **Use statements** - Track module dependencies for AST building
//!
//! ## Variadic Expansion
//!
//! Items marked with variadic attributes are expanded into multiple versions:
//!
//! ```rust,ignore
//! // Input:
//! #[cuda_tile::variadic_struct(N=4)]
//! pub struct Tile<E, const D: [i32; N]> { }
//!
//! // Output: Tile_1, Tile_2, Tile_3, Tile_4 structs
//! ```
//!
//! ## AST Generation
//!
//! For each module, a `_module_asts()` function is generated that returns a vector
//! of MLIR AST modules. This is used during compilation to build the CUDA code.
//!
//! ## Kernel Launchers
//!
//! For each `#[entry]` function, an async launcher is generated that:
//! - Compiles the kernel to CUDA
//! - Manages kernel invocation
//! - Handles async execution
//! - Provides type-safe parameter passing

use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use proc_macro2::Ident;
use proc_macro2::{LineColumn, Span, TokenStream as TokenStream2};
use quote::{quote, ToTokens};
use std::collections::HashSet;
use std::path::PathBuf;
use std::{env, fs};

use syn::{
    parse_file, parse_macro_input, parse_quote, AngleBracketedGenericArguments, GenericArgument,
    GenericParam, Item, ItemFn, ItemImpl, ItemMod, ItemStruct, ItemTrait, ItemUse, Macro, Path,
    UseTree,
};

use crate::error::{Error, SpannedError};
use crate::kernel_launcher_generator::generate_kernel_launcher;
use crate::rewrite_variadics::*;
use crate::validate_dsl_syntax::validate_entry_point_parameters;
use cutile_compiler::syn_utils::*;

fn line_column_to_offset(source: &str, loc: LineColumn) -> Option<usize> {
    let mut line_start = 0usize;
    let mut current_line = 1usize;

    for line in source.split_inclusive('\n') {
        if current_line == loc.line {
            let column_offset = byte_offset_for_char_column(line, loc.column)?;
            return Some(line_start + column_offset);
        }
        line_start += line.len();
        current_line += 1;
    }

    if current_line == loc.line {
        let tail = &source[line_start..];
        let column_offset = byte_offset_for_char_column(tail, loc.column)?;
        return Some(line_start + column_offset);
    }

    None
}

fn byte_offset_for_char_column(line: &str, column: usize) -> Option<usize> {
    if column == 0 {
        return Some(0);
    }

    if column == line.chars().count() {
        return Some(line.len());
    }

    line.char_indices().nth(column).map(|(idx, _)| idx)
}

fn source_slice_from_file(path: &str, start: LineColumn, end: LineColumn) -> Option<String> {
    let source = fs::read_to_string(path).ok()?;
    let start_offset = line_column_to_offset(&source, start)?;
    let end_offset = line_column_to_offset(&source, end)?;
    source.get(start_offset..end_offset).map(str::to_string)
}

/// Returns the path to the CUDA tile AST module.
///
/// This is used throughout the code generation to reference AST types.
pub fn get_ast_path(tile_rust_crate_root: &Ident) -> Path {
    let s = format!("{tile_rust_crate_root}::cutile_compiler::ast");
    syn::parse::<Path>(s.parse().unwrap()).unwrap()
}

/// Returns the identifier for the module AST builder function.
///
/// Each module generates a function with this name that builds its AST.
pub fn get_asts_ident() -> Ident {
    Ident::new("_module_asts", Span::call_site())
}

/// Main entry point for the module macro.
///
/// Transforms a Rust module containing GPU kernel code into:
/// - Concrete Rust functions (possibly expanded from variadics)
/// - MLIR AST builder functions
/// - Async kernel launcher functions
///
/// ## Processing Pipeline
///
/// 1. Parse module attributes (`core`, `tile_rust_crate`)
/// 2. Iterate through module items
/// 3. For each item:
///    - Validate syntax (for functions)
///    - Generate AST representation
///    - Handle variadic expansion if needed
///    - Generate kernel launcher if `#[entry]` function
/// 4. Generate module AST builder function
/// 5. Emit expanded code with all dependencies
///
/// ## Attributes
///
/// - `core=true` - This is a core DSL module (allows more item types)
/// - `tile_rust_crate=true` - This module is within the cutile crate
///
/// ## Generated Structure
///
/// ```rust,ignore
/// pub mod my_module {
///     // Imports and dependencies
///     use cutile_compiler::ast;
///     use cutile::tile_async::*;
///
///     // Module AST builder
///     pub fn _module_asts() -> Vec<Module> { ... }
///
///     // Concrete items (functions, structs, etc.)
///     pub fn my_kernel(...) { ... }
///
///     // Kernel launchers (for #[entry] functions)
///     pub fn my_kernel_apply(...) -> impl DeviceOperation { ... }
/// }
/// ```
// TODO (hme): Prevent reserved names from being used.
// TODO (hme): Validate supported modules.
pub fn module(attributes: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attributes as SingleMetaList);
    let is_core = attrs.parse_bool("core").unwrap_or(false);
    let is_tile_rust_crate = attrs.parse_bool("tile_rust_crate").unwrap_or(false);
    let tile_rust_crate_root = Ident::new(
        if is_tile_rust_crate {
            "crate"
        } else {
            "cutile"
        },
        Span::call_site(),
    );

    // Get raw item source as a fallback for source text
    // capture.  The primary path uses `Span::source_text()` (which preserves
    // comments), but if that is unavailable we fall back to
    // `TokenStream::to_string()` (which strips comments).
    let raw_item_source = item.to_string();

    let mut module_item = parse_macro_input!(item as ItemMod);
    module_item.attrs = attrs.into();

    match module_inner(
        &module_item,
        is_core,
        &tile_rust_crate_root,
        raw_item_source,
    ) {
        Ok(ts) => ts.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Fallible inner implementation of the `module` macro.
fn module_inner(
    module_item: &ItemMod,
    is_core: bool,
    tile_rust_crate_root: &Ident,
    raw_item_source: String,
) -> Result<TokenStream2, Error> {
    let mut ast_content: Vec<Item> = vec![];
    let Some(content) = &module_item.content else {
        return module_item.err("Non-empty module expected.");
    };
    let mut concrete_items: Vec<TokenStream2> = vec![];
    let name = &module_item.ident;
    let mut module_ast_calls: Vec<String> = vec![];
    let mut entry_functions: Vec<TokenStream2> = vec![];

    for item in &content.1 {
        match item {
            syn::Item::Use(use_item) => {
                concrete_items.push(use_item.to_token_stream());
                // Include module_ast dependency as part of the export.
                if !is_core {
                    // println!("{use_item:#?}");
                    let mut use_tree = &use_item.tree;
                    let mut module_ast_use_path = vec![];
                    while let UseTree::Path(path) = use_tree {
                        let path_ident_str = path.ident.to_string();
                        module_ast_use_path.push(path_ident_str);
                        use_tree = &path.tree;
                    }
                    let module_ast_call_str = format!(
                        "{}::{}()",
                        module_ast_use_path.last().unwrap(),
                        get_asts_ident()
                    );
                    module_ast_calls.push(module_ast_call_str);
                    let module_ast_use_path_str =
                        format!("use {};", module_ast_use_path.join("::"));
                    let module_ast_use_path_item =
                        syn::parse::<ItemUse>(module_ast_use_path_str.parse().unwrap()).unwrap();
                    concrete_items.push(module_ast_use_path_item.to_token_stream());
                }
            }
            syn::Item::Fn(function_item) => {
                let entry_attrs = get_meta_list(
                    format!("{} :: entry", tile_rust_crate_root).as_str(),
                    &function_item.attrs,
                );
                if entry_attrs.is_some() {
                    entry_functions.push(kernel_launcher(name, function_item)?);
                };
                ast_content.push(Item::Fn(function_item.clone()));
                concrete_items.push(function(function_item.clone(), tile_rust_crate_root)?);
            }
            syn::Item::Struct(struct_item) => {
                ast_content.push(Item::Struct(struct_item.clone()));
                let item_clone = struct_item.clone();
                concrete_items.push(structure(item_clone)?.into());
            }
            syn::Item::Trait(trait_item) => {
                if !is_core {
                    return trait_item.err("Unsupported item type in non-core module: trait definitions are only allowed in core modules.");
                }
                ast_content.push(Item::Trait(trait_item.clone()));
                let item_clone = trait_item.clone();
                concrete_items.push(trait_(item_clone)?.into());
            }
            syn::Item::Type(type_item) => {
                concrete_items.push(type_item.to_token_stream());
            }
            syn::Item::Impl(impl_item) => {
                if !is_core {
                    return impl_item.err("Unsupported item type in non-core module: impl blocks are only allowed in core modules.");
                }
                ast_content.push(Item::Impl(impl_item.clone()));
                let item_clone = impl_item.clone();
                concrete_items.push(implementation(item_clone)?.into());
            }
            syn::Item::Macro(macro_item) => {
                if !is_core {
                    return macro_item.err("Unsupported item type in non-core module: macro invocations are only allowed in core modules.");
                }
                ast_content.push(Item::Macro(macro_item.clone()));
                let item_clone = macro_item.clone();
                concrete_items.push(item_clone.to_token_stream());
            }
            other => {
                return other.err("Unsupported item type in module.");
            }
        }
    }
    let ast_path = get_ast_path(tile_rust_crate_root);
    let ast_module_item: ItemMod = module_item.clone();
    let ast_module_tokens = module_asts(
        ast_module_item,
        module_ast_calls,
        tile_rust_crate_root,
        raw_item_source,
    );
    let res = if entry_functions.is_empty() {
        quote! {
            pub mod #name {
                #![allow(nonstandard_style)]
                #![allow(dead_code)]
                #![allow(unused_variables)]
                // Module asts and generated type data.
                use #ast_path;
                #ast_module_tokens
                #(#concrete_items)*
            }
        }
    } else {
        quote! {
            pub mod #name {
                #![allow(dead_code)]
                // Entry point dependencies.
                // Use of this macro requires cutile,
                // so all dependencies should be imported relative to cutile.
                use std::{iter::zip, future::{Future, IntoFuture}, collections::HashMap, sync::Arc};
                use #tile_rust_crate_root::error::{*};
                use #tile_rust_crate_root::WithDType;
                use #tile_rust_crate_root::{tensor};
                use #tile_rust_crate_root::tile_kernel::{*};
                use #tile_rust_crate_root::cuda_async::error::{*};
                use #tile_rust_crate_root::cuda_async::scheduling_policies::SchedulingPolicy;
                use #tile_rust_crate_root::cuda_core::{CudaContext, CudaFunction, CudaModule, CudaStream, DriverError, LaunchConfig};
                // use #tile_rust_crate_root::cutile_compiler::cuda_tile::ModuleOperation;
                // use #tile_rust_crate_root::cutile_compiler::compiler::{CUDATileModules, CUDATileFunctionCompiler};
                // Module asts and generated type data.
                use #ast_path;
                #ast_module_tokens
                #(#concrete_items)*
                // Entry point code.
                #(#entry_functions)*
            }
        }
    };
    Ok(res)
}

/// Processes trait definitions.
///
/// Handles trait definitions that may be variadic (rank-polymorphic). Traits marked
/// with `#[cuda_tile::variadic_trait]` are expanded into multiple versions.
///
/// ## Variadic Traits
///
/// Variadic traits are expanded into separate traits for each rank (1D through 4D):
///
/// ```rust,ignore
/// // Input:
/// #[cuda_tile::variadic_trait(N=4)]
/// pub trait MyTrait<const D: [i32; N]> {
///     fn method(&self) -> Tile<f32, D>;
/// }
///
/// // Output: MyTrait_1, MyTrait_2, MyTrait_3, MyTrait_4
/// ```
///
/// ## Const Generic Desugaring
///
/// For non-variadic traits, const generic array parameters are desugared to make
/// them compatible with the MLIR type system.
pub fn trait_(mut item: ItemTrait) -> Result<TokenStream, Error> {
    // println!("implementation {ident}: {attributes:#?}");
    let attributes = get_meta_list("cuda_tile :: variadic_trait", &item.attrs);
    let is_unchecked = get_meta_list("cuda_tile :: unchecked", &item.attrs);
    if is_unchecked.is_some() {
        return Ok(quote! {}.into());
    }
    clear_attributes(
        HashSet::from(["cuda_tile :: variadic_trait", "cuda_tile :: ty"]),
        &mut item.attrs,
    );
    let res = match attributes {
        Some(attributes) => match attributes.name_as_str().unwrap() {
            "cuda_tile :: variadic_trait" => {
                let items = variadic_trait(&attributes, item)?;
                quote! {
                    #(#items)*
                }
            }
            _ => {
                let item = desugar_trait_cgas(&item)?;
                quote! { #item }
            }
        },
        None => {
            let item = desugar_trait_cgas(&item)?;
            quote! { #item }
        }
    };
    Ok(res.into())
}

/// Processes trait and inherent implementations.
///
/// Handles `impl` blocks that may be variadic. Implementations marked with
/// `#[cuda_tile::variadic_impl]` are expanded into multiple rank-specific versions.
///
/// ## Variadic Implementations
///
/// Variadic implementations are expanded to match their corresponding variadic types:
///
/// ```rust,ignore
/// // Input:
/// #[cuda_tile::variadic_impl(N=4)]
/// impl<E: ElementType, const D: [i32; N]> Tile<E, D> {
///     pub fn shape(&self) -> Shape<D> { ... }
/// }
///
/// // Output: Impl blocks for Tile_1, Tile_2, Tile_3, Tile_4
/// ```
///
/// ## Operator Implementations
///
/// Operator traits (`Add`, `Sub`, etc.) are also expanded variadically:
///
/// ```rust,ignore
/// #[cuda_tile::variadic_impl(N=4)]
/// impl<E, const D: [i32; N]> ops::Add<Tile<E, D>> for Tile<E, D> {
///     type Output = Tile<E, D>;
///     fn add(self, rhs: Tile<E, D>) -> Tile<E, D> { ... }
/// }
/// ```
pub fn implementation(mut item: ItemImpl) -> Result<TokenStream, Error> {
    // println!("implementation {ident}: {attributes:#?}");
    let attributes = get_meta_list("cuda_tile :: variadic_impl", &item.attrs);
    let is_unchecked = get_meta_list("cuda_tile :: unchecked", &item.attrs);
    if is_unchecked.is_some() {
        return Ok(quote! {}.into());
    }
    clear_attributes(
        HashSet::from([
            "cuda_tile :: variadic_trait_impl",
            "cuda_tile :: variadic_impl",
            "cuda_tile :: ty",
        ]),
        &mut item.attrs,
    );
    let res = match attributes {
        Some(attributes) => match attributes.name_as_str().unwrap() {
            "cuda_tile :: variadic_impl" => {
                let items = variadic_impl(&attributes, item)?;
                quote! {
                    #(#items)*
                }
            }
            _ => {
                let item = desugar_impl_cgas(&item)?;
                quote! { #item }
            }
        },
        None => {
            let item = desugar_impl_cgas(&item)?;
            quote! { #item }
        }
    };
    Ok(res.into())
}

/// Processes struct definitions.
///
/// Handles struct definitions that may be variadic. Structs marked with
/// `#[cuda_tile::variadic_struct]` are expanded into multiple rank-specific versions.
///
/// ## Variadic Structs
///
/// The most common variadic structs in cuTile Rust are the core types:
///
/// ```rust,ignore
/// // Input:
/// #[cuda_tile::variadic_struct(N=4, constructor="new")]
/// pub struct Tile<E: ElementType, const D: [i32; N]> {
///     _type: PhantomData<E>
/// }
///
/// // Output: Tile_1, Tile_2, Tile_3, Tile_4
/// // Plus optional constructor implementations if specified
/// ```
///
/// ## Constructor Generation
///
/// If `constructor="name"` is specified in the attribute, a constructor method
/// is automatically generated for each expanded struct.
///
/// ## Const Generic Desugaring
///
/// For non-variadic structs, const generic array parameters are desugared to
/// be compatible with the type system.
pub fn structure(mut item: ItemStruct) -> Result<TokenStream, Error> {
    let attributes = get_meta_list("cuda_tile :: variadic_struct", &item.attrs);
    clear_attributes(
        HashSet::from(["cuda_tile :: variadic_struct", "cuda_tile :: ty"]),
        &mut item.attrs,
    );
    // println!("structure {ident}: {attributes:#?}");
    let res = match attributes {
        Some(attributes) => match attributes.name_as_str().unwrap() {
            "cuda_tile :: variadic_struct" => {
                let items = variadic_struct(&attributes, item)?;
                let structs = items.iter().map(|item| item.0.clone()).collect::<Vec<_>>();
                let maybe_impls = items
                    .iter()
                    .filter(|item| item.1.is_some())
                    .collect::<Vec<_>>();
                let impls = maybe_impls
                    .iter()
                    .map(|item| item.1.clone().unwrap())
                    .collect::<Vec<_>>();
                quote! {
                    #(#structs)*
                    #(#impls)*
                }
            }
            _ => {
                let item = desugar_structure_cgas(&item)?;
                quote! { #item }
            }
        },
        None => {
            let item = desugar_structure_cgas(&item)?;
            quote! { #item }
        }
    };
    Ok(res.into())
}

/// Processes function definitions.
///
/// Transforms GPU kernel functions and DSL helper functions. Handles:
/// - Variadic operations (rank-polymorphic functions)
/// - Compiler operations (builtin operations like `cast`, `convert`)
/// - Entry point validation
/// - AST building code generation
///
/// ## Variadic Operations
///
/// Functions marked with `#[cuda_tile::variadic_op]` are expanded:
///
/// ```rust,ignore
/// // Input:
/// #[cuda_tile::variadic_op(N=4)]
/// pub fn load_tile<E: ElementType, const S: [i32; N]>(y: &mut Tensor<E, S>) -> Tile<E, S> {
///     // ...
/// }
///
/// // Output: load_tile_1, load_tile_2, load_tile_3, load_tile_4
/// ```
///
/// ## Compiler Operations
///
/// Functions marked with `#[cuda_tile::compiler_op]` are treated as compiler built-ins
/// and generate appropriate MLIR operations.
///
/// ## Entry Points
///
/// Functions marked with `#[entry]` are validated and have async launchers generated.
pub fn function(mut item: ItemFn, tile_rust_crate_root: &Ident) -> Result<TokenStream2, Error> {
    if get_meta_list_by_last_segment("entry", &item.attrs).is_some() {
        validate_entry_point_parameters(&item)?
    }
    let attributes = get_meta_list("cuda_tile :: variadic_op", &item.attrs);
    clear_attributes(
        HashSet::from([
            "cuda_tile :: variadic_op",
            "cuda_tile :: op",
            "cuda_tile :: compiler_op",
        ]),
        &mut item.attrs,
    );
    clear_attributes(
        HashSet::from([format!("{} :: entry", tile_rust_crate_root).as_str()]),
        &mut item.attrs,
    );
    let concrete_items = match attributes {
        Some(attributes) => match attributes.name_as_str().unwrap() {
            "cuda_tile :: variadic_op" => variadic_op(&attributes, item.clone())?,
            _ => vec![desugar_function_cgas(&item)?],
        },
        None => vec![desugar_function_cgas(&item)?],
    };
    let result = quote! {
        #(#concrete_items)*
    };
    Ok(result)
}

/// Generates the complete async kernel launcher struct and implementations.
///
/// Creates a launcher struct that implements `TileKernel`, `DeviceOperation`, and
/// `IntoFuture` for a kernel entry point. This enables the ergonomic async API
/// for launching GPU kernels.
///
/// ## Parameters
///
/// - `module_ident`: The module identifier containing the kernel
/// - `item`: The kernel function AST (marked with `#[entry]`)
///
/// ## Returns
///
/// Token stream containing:
/// 1. The launcher struct definition (e.g., `pub struct MyKernel<T, DI> { ... }`)
/// 2. `TileKernel` impl (provides `.grid()`, `.const_grid()`, `.generics()` methods)
/// 3. `DeviceOperation` impl (provides `.execute()` for actual kernel launch)
/// 4. `IntoFuture` impl (enables `.await` syntax)
///
/// ## Generated API
///
/// For a kernel `fn my_kernel<T>(x: &mut Tensor<T, [128]>)`, generates:
/// ```rust,ignore
/// let result = MyKernel::launch(input_data)
///     .grid((num_blocks, 1, 1))
///     .await;
/// ```
///
/// ## Entry Attributes
///
/// Respects `#[entry(print_ir = true)]` to print the generated launcher code.
pub fn kernel_launcher(module_ident: &Ident, item: &ItemFn) -> Result<TokenStream2, Error> {
    let module_name = module_ident.to_string();
    let function_name = item.sig.ident.to_string();
    let function_entry_name = format!("{}_entry", function_name);
    let launcher_name = function_name.to_case(Case::UpperCamel);
    let launcher_args_name = format!("{}Args", launcher_name);
    let unsafety = item.sig.unsafety;

    let (required_generics, (launcher_args_type, launcher_type_def), device_op_impl) =
        generate_kernel_launcher(
            item,
            &module_name,
            &function_name,
            &function_entry_name,
            &launcher_name,
            &launcher_args_name,
        )?;

    let launcher_ident = Ident::new(launcher_name.as_str(), Span::call_site());
    let _launcher_args_ident = Ident::new(launcher_args_name.as_str(), Span::call_site());

    let generic_params = required_generics.get_required_generics();
    let generic_args = required_generics.get_generic_args();

    let device_op_param: GenericParam =
        parse_quote! { DI: DeviceOperation<Output=#launcher_args_type> };
    let device_op_arg: GenericArgument = parse_quote! { DI };

    // Struct
    let mut struct_generics = generic_params.clone();
    struct_generics.params.push(device_op_param.clone());
    let mut struct_args = generic_args.clone();
    struct_args.args.push(device_op_arg.clone());

    // impl TileKernel
    let tile_kernel_impl_type_params = struct_generics.clone();
    let tile_kernel_type_args: AngleBracketedGenericArguments =
        parse_quote! { <#launcher_args_type, #device_op_arg> };

    // impl DeviceOperation
    let _device_operation_impl_type_params = struct_generics.clone();

    // impl IntoFuture
    let into_future_impl_type_params = struct_generics.clone();

    let result = quote! {

        #launcher_type_def

        #[derive(Debug)]
        pub struct #launcher_ident #struct_generics {
            _const_grid: bool,
            _grid: (u32, u32, u32),
            input: Option<DI>,
            function_generics: Option<Vec<String>>
        }

        impl #tile_kernel_impl_type_params #launcher_ident #struct_args {
            pub #unsafety fn launch(input: DI) -> Self {
                Self {
                    _const_grid: false,
                    _grid: (0, 0, 0),
                    input: Some(input),
                    function_generics: None
                }
            }
        }

        // TileKernel impl provides things like the compile method.
        impl #tile_kernel_impl_type_params TileKernel #tile_kernel_type_args for #launcher_ident #struct_args {
            fn grid(mut self, grid: (u32, u32, u32)) -> Self {
                self._grid = grid;
                self._const_grid = false;
                self
            }
            fn const_grid(mut self, grid: (u32, u32, u32)) -> Self {
                self._grid = grid;
                self._const_grid = true;
                self
            }
            fn get_launch_grid(&self) -> (u32, u32, u32) {
                self._grid
            }
            fn generics(mut self, generics: Vec<String>) -> Self {
                self.function_generics = Some(generics);
                self
            }
        }
        // Easily convert into a future.
        impl #into_future_impl_type_params IntoFuture for #launcher_ident #struct_args {
            type Output = Result<#launcher_args_type, DeviceError>;
            type IntoFuture = DeviceFuture<#launcher_args_type, #launcher_ident #struct_args>;
            fn into_future(self) -> Self::IntoFuture {
                match with_default_device_policy(|policy| policy.schedule(self)) {
                    Ok(Ok(future)) => future,
                    Ok(Err(e)) => DeviceFuture::failed(e),
                    Err(e) => DeviceFuture::failed(e),
                }
            }
        }

        // Implements DeviceOperation, along with sync/async launcher functions.
        #device_op_impl
    };

    let Some(_entry_attrs) = get_meta_list_by_last_segment("entry", &item.attrs) else {
        return item.sig.ident.err(&format!(
            "Unexpected entry point {function_name}: Missing entry annotation."
        ));
    };

    if let Ok(dir) = env::var("DUMP_KERNEL_LAUNCHER_DIR") {
        let file = parse_file(&result.to_string()).expect("Failed to parse file.");
        let filename = format!("{module_name}_{function_name}_launcher.rs");
        let path = PathBuf::from(dir).join(filename);
        let contents = file_item_string_pretty(&file);
        fs::write(path.clone(), contents).unwrap_or_else(|_| panic!("Failed to write {path:?}"));
        // Writes the string as bytes
    }
    Ok(result)
}

/// Generates the module AST builder function.
/// Creates a function that returns the AST representations of all kernels and
/// functions defined in the module. This AST is used by the MLIR compiler to
/// generate GPU code.
///
/// ## Parameters
///
/// - `item`: The module AST
/// - `module_ast_calls`: Vector of strings containing calls to AST builder functions
///   for each kernel/function in the module (e.g., `["my_kernel_asts()", "helper_asts()"]`)
///
/// ## Returns
///
/// Token stream containing a function like:
/// ```rust,ignore
/// pub fn module_name_asts() -> Vec<cuda_ast::Ast> {
///     vec![my_kernel_asts(), helper_asts()]
/// }
/// ```
///
/// ## Purpose
///
/// This function aggregates all ASTs from a module, enabling the module to be
/// compiled as a unit. The ASTs are later passed to the MLIR compiler for code
/// generation.
///
/// ## Source Location Tracking
///
/// At proc macro expansion time, `proc_macro2` spans carry real file / line /
/// column information (on nightly with the `span-locations` feature).  We
/// exploit this to recover **exact** source locations for *every* node in the
/// syn AST at JIT compile time, using the following scheme:
///
/// 1. Construct a `Span` covering the entire `ItemMod` (from the `mod`
///    keyword to the closing `}`) and call `Span::source_text()` to obtain
///    the **verbatim** original source text — whitespace, newlines, comments,
///    and all.  (`TokenStream::to_string()` strips comments, so we must use
///    `source_text()` instead.)
/// 2. Record the **span base** – `(file, base_line, base_col)` – from the
///    module's opening token via `Span::file()` and `Span::start()`.
/// 3. At runtime, feed the source text to `syn::parse_str` instead of
///    re-quoting.  Because the string is character-for-character identical to
///    the original source, the resulting spans have line/column numbers that
///    map 1-to-1 with the original file layout.
/// 4. Any runtime span can then be resolved to an absolute position via:
///    ```text
///    abs_line = base_line + (span_line − 1)
///    abs_col  = if span_line == 1 { base_col + span_col } else { span_col }
///    ```
///
/// This gives exact file / line / column for every node – statements,
/// expressions, sub-expressions, individual tokens – without requiring any
/// up-front walk or key-based lookup table.
///
/// ### Fallback
///
/// If `Span::source_text()` is unavailable (e.g. on a stable compiler, or
/// when the span doesn't map to real source), we fall back to
/// `TokenStream::to_string()`.  This produces comment-free text, so line
/// numbers may be shifted earlier by the number of stripped comment lines.
pub fn module_asts(
    item: ItemMod,
    module_ast_calls: Vec<String>,
    tile_rust_crate_root: &Ident,
    raw_item_source: String,
) -> TokenStream2 {
    // TODO (hme): Double check Rust will handle circular dependencies and report them.
    let ast_path = get_ast_path(tile_rust_crate_root);
    let name_string = item.ident.to_string();
    let vec_expr_str = format!("vec![{}]", module_ast_calls.join(","));
    let vec_expr = syn::parse::<Macro>(vec_expr_str.parse().unwrap()).unwrap();
    let asts_ident = get_asts_ident();

    // --- Capture span base at proc macro time --------------------------------
    //
    // The first token's span tells us where the module lives in the original
    // source file.  We record (file, line, col) so that at runtime we can
    // convert any string-relative span position produced by `syn::parse_str`
    // into an absolute file:line:col.
    // Use the `mod` keyword's span as the anchor — NOT `item.span()`.
    //
    // `item.span()` (via syn's `Spanned`) joins the spans of *all* child
    // tokens including `attrs`.  Because `module_item.attrs` was replaced
    // earlier with the parsed proc-macro attributes (whose spans point at
    // the attribute invocation site), `item.span()` may start at the
    // `#[cutile::module(...)]` attribute rather than the `mod` keyword.
    // That causes `source_text()` to return the attribute text instead of
    // the module body.
    //
    // The `mod_token` span always points to the `mod` keyword in the
    // original source, but we prefer the visibility span (e.g. `pub`) when
    // present since it comes first and `source_text()` must cover the
    // complete `pub mod foo { … }` text for `syn::parse_str` to succeed.
    let item_start_span = match &item.vis {
        syn::Visibility::Public(vis_pub) => vis_pub.span,
        syn::Visibility::Restricted(vis_r) => vis_r.pub_token.span,
        syn::Visibility::Inherited => item.mod_token.span,
    };
    let source_file = item_start_span.file();
    let base_line = item_start_span.start().line;
    let base_col = item_start_span.start().column;

    // --- Source text ----------------------------------------------------------
    //
    // We need the **verbatim** source text of the module — including comments,
    // whitespace, and all — so that when `syn::parse_str` re-parses it at
    // runtime, the resulting line/column numbers map 1-to-1 with the original
    // file layout.
    //
    // `Span::source_text()` returns exactly this: the slice of the original
    // source file that the span covers, comments and all.  We construct a
    // span covering the entire `ItemMod` by joining the span of the opening
    // `mod` keyword with the span of the closing `}`.
    //
    // If `source_text()` is unavailable (for example on a stable compiler),
    // we reconstruct the exact source slice from the original file using the
    // span start/end positions. Only if that fails do we fall back to
    // `TokenStream::to_string()`.
    let source_text = {
        let full_span = item
            .content
            .as_ref()
            .and_then(|(brace, _)| item_start_span.join(brace.span.close()));
        let file_slice = item.content.as_ref().and_then(|(brace, _)| {
            source_slice_from_file(
                &source_file.to_string(),
                item_start_span.start(),
                brace.span.close().end(),
            )
        });

        full_span
            .and_then(|sp| sp.source_text())
            .or(file_slice)
            .unwrap_or(raw_item_source)
    };

    let result = quote! {
        pub fn #asts_ident() -> Vec<#ast_path::Module> {
            use #ast_path::syn;

            // Re-parse the source text that was captured at proc macro time.
            // When `Span::source_text()` was available, this is the verbatim
            // original source (with comments), so `syn::parse_str` produces
            // spans whose line/column correspond 1-to-1 with the original
            // file layout. When `source_text()` is unavailable, the macro
            // reconstructs the exact source slice from the original file so
            // stable builds preserve line numbers too.
            let source_text: &str = #source_text;
            let parsed_mod: syn::ItemMod = syn::parse_str(source_text)
                .expect("module_asts: failed to re-parse captured source text");

            let span_base = #ast_path::SpanBase::new(
                #source_file.to_string(),
                #base_line,
                #base_col,
            );

            let this_ast = #ast_path::Module::with_span_base(
                #name_string,
                parsed_mod,
                span_base,
            );
            let mut module_asts: Vec<#ast_path::Module> = vec![];
            let other_module_asts_asts: Vec<Vec<#ast_path::Module>> = #vec_expr;
            for other_module_asts in other_module_asts_asts {
                for module_ast in other_module_asts {
                    module_asts.push(module_ast);
                }
            }
            module_asts.push(this_ast);
            module_asts
        }
    };
    result
}
