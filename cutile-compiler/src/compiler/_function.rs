/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Core compiler struct and entry point for compiling a single CUDA Tile function.
//! Expression, block, type, inline, intrinsic, binary_op, cuda_tile_op, derive,
//! constant, and assume compilation are delegated to `compile_*` submodules.

use crate::ast::{SourceLocation, SpanBase};
use crate::bounds::Bounds;
use crate::compiler::_module::CUDATileModules;
pub use crate::compiler::_type::*;
pub use crate::compiler::_value::*;
use crate::compiler::utils::{
    named_flag_attr, named_str_attr, named_type_attr, parse_named_attr, OptimizationHints,
};
use crate::context_all;
use crate::cuda_tile;
use crate::cuda_tile::ModuleOperation;
use crate::error::JITError;
use crate::error::SpannedJITError;
use crate::generics::{GenericVars, TypeInstance};
use crate::kernel_entry_generator::generate_entry_point;
use crate::kernel_naming::KernelNaming;
use crate::syn_utils::*;
use crate::types::*;
use cuda_core::Validator;
use cuda_tile_rs::operation_parse;
use melior::ir::attribute::StringAttribute;
use melior::ir::operation::{OperationBuilder, OperationLike};
use melior::ir::r#type::FunctionType;
use melior::ir::{
    self, Attribute, Block, BlockLike, Identifier, Location, Operation, Region, RegionLike, Value,
};
use melior::Context;

use quote::ToTokens;
use std::any::type_name;
use std::collections::HashMap;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{Expr, ItemFn, Type};

use anyhow::{Context as AnyhowContext, Result};

/// Minimum remaining stack space before growing (1 MiB).
pub(crate) const STACK_RED_ZONE: usize = 1 * 1024 * 1024;
/// Size of each new stack segment when growth is needed (10 MiB).
pub(crate) const STACK_GROW_SIZE: usize = 10 * 1024 * 1024;

/// Compiles a single Rust function into a CUDA Tile MLIR module.
pub struct CUDATileFunctionCompiler<'m> {
    pub(crate) context: Context,
    // ASTs
    pub(crate) modules: &'m CUDATileModules,
    pub(crate) module_name: String,
    pub(crate) function_name: String,
    // Function to compile.
    pub(crate) function: &'m ItemFn,
    pub(crate) entry: ItemFn,
    // JIT args.
    pub(crate) entry_attrs: EntryAttrs,
    pub(crate) const_grid: Option<(u32, u32, u32)>,
    pub(crate) gpu_name: String,
    pub(crate) optimization_hints: OptimizationHints,
    pub(crate) stride_args: HashMap<String, Vec<i32>>,
    pub(crate) generic_vars: GenericVars,
    // Validate static params, like shape of tensors.
    pub(crate) validator: Validator,
    pub(crate) module_name_stack: Vec<String>,
}

/// Parsed attributes from the `#[cuda_tile::entry(...)]` annotation on a kernel function.
pub struct EntryAttrs {
    entry_attrs: SingleMetaList,
}

impl EntryAttrs {
    pub(crate) fn get_entry_arg_expr(&self, name: &str) -> Option<&Expr> {
        self.entry_attrs.parse_custom_expr(name)
    }
    pub(crate) fn get_entry_arg_bool(&self, name: &str) -> bool {
        self.entry_attrs.parse_bool(name).unwrap_or(false)
    }
}

impl<'m, 'c> CUDATileFunctionCompiler<'m> {
    pub fn new(
        modules: &'m CUDATileModules,
        module_name: &str,
        function_name: &str,
        function_generic_args: &[String],
        stride_args: &[(&str, &[i32])],
        const_grid: Option<(u32, u32, u32)>,
        gpu_name: String,
    ) -> Result<Self, JITError> {
        if !modules.modules.contains_key(module_name) {
            return Err(JITError::Generic(format!(
                "Undefined module: {module_name}"
            )));
        }
        let kernel_naming = KernelNaming::new(function_name);

        let (_, function) = modules
            .functions
            .get(kernel_naming.public_name())
            .with_context(|| format!("Undefined function: {function_name}"))?;

        let entry_attrs =
            get_meta_list_by_last_segment("entry", &function.attrs).ok_or_else(|| {
                modules
                    .resolve_span(module_name, &function.span())
                    .jit_error(&format!(
                    "function `{function_name}` is missing a required `#[entry(...)]` attribute"
                ))
            })?;
        let entry_attrs = EntryAttrs { entry_attrs };

        if entry_attrs.get_entry_arg_bool("unchecked_accesses") && function.sig.unsafety.is_none() {
            return modules
                .resolve_span(module_name, &function.span())
                .jit_error_result(
                    "kernel must be declared `unsafe` when `unchecked_accesses` is enabled",
                );
        }

        let optimization_hints = match entry_attrs.get_entry_arg_expr("optimization_hints") {
            Some(hints_expr) => OptimizationHints::parse(hints_expr, gpu_name.clone())?,
            None => OptimizationHints::empty(),
        };

        let stride_args: HashMap<String, Vec<i32>> = stride_args
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_vec()))
            .collect::<HashMap<_, _>>();

        let generic_vars = GenericVars::from_flat(&function.sig.generics, function_generic_args)?;

        let (entry, validator) = generate_entry_point(
            &function,
            &generic_vars,
            &stride_args,
            &modules.primitives,
            &optimization_hints,
        )?;

        if modules
            .functions
            .get(kernel_naming.entry_name().as_str())
            .is_some()
        {
            return modules
                .resolve_span(module_name, &function.span())
                .jit_error_result(&format!(
                    "Entry point namespace collision: {}",
                    kernel_naming.entry_name()
                ));
        }

        if entry_attrs.get_entry_arg_bool("print_ir") {
            println!("GENERATED ENTRY POINT: {module_name}::{function_name}");
            println!("{}", item_string_pretty(&entry.clone().into()));
            println!();
        }

        Ok(CUDATileFunctionCompiler {
            context: context_all(),
            modules,
            module_name: module_name.to_string(),
            function_name: function_name.to_string(),
            entry_attrs,
            const_grid,
            gpu_name,
            optimization_hints,
            function,
            entry,
            validator,
            generic_vars,
            stride_args,
            module_name_stack: vec![module_name.to_string()],
        })
    }

    pub fn compile(&self) -> Result<ModuleOperation<'_>, JITError> {
        let module_name = &self.module_name;
        let function_name = &self.function_name;
        let fn_item = self.function;
        // return self.jit_error_result(&self.function.span(), &format!("Error"));

        if self.entry_attrs.get_entry_arg_bool("print_ir") {
            println!("COMPILING FUNCTION: {module_name}::{function_name}");
            println!("{}", item_string_pretty(&fn_item.clone().into()));
            println!();
        }

        let module_block = Block::new(&[]);
        module_block.append_operation(self.compile_function(
            &self.entry,
            &self.generic_vars,
            &self.stride_args,
        )?);

        let location = self.function_location();
        let module_op = cuda_tile::ModuleOperationBuilder::new(&self.context, location)
            .body({
                let region = Region::new();
                region.append_block(module_block);
                region
            })
            .sym_name(StringAttribute::new(&self.context, module_name))
            .build();
        if module_op.as_operation().verify() {
            Ok(module_op)
        } else {
            return self.jit_error_result(
                &self.function.span(),
                &format!(
                    "Failed to verify module {}",
                    module_op.as_operation().to_string()
                ),
            );
        }
    }

    /// Convert a [`SourceLocation`] into an MLIR `Location`.
    ///
    /// If the source location is known (has a real file path and line number),
    /// this produces `Location::new(ctx, file, line, col)` so that MLIR errors
    /// and printed IR reference the user's original source code.  Otherwise it
    /// falls back to `Location::unknown`.
    pub(crate) fn mlir_location(&self, src: &SourceLocation) -> Location<'_> {
        if src.is_known() {
            Location::new(&self.context, &src.file, src.line, src.column)
        } else {
            Location::unknown(&self.context)
        }
    }

    /// Return the [`SpanBase`] for the module currently being compiled,
    /// or a default unknown base if none was captured.
    pub(crate) fn span_base(&self) -> SpanBase {
        let current_module = &self.module_name_stack[0];
        self.modules
            .get_span_base(current_module)
            .cloned()
            .unwrap_or_default()
    }

    pub(crate) fn jit_error(&self, span: &proc_macro2::Span, error_message: &str) -> JITError {
        self.resolve_span(span).jit_error(error_message)
    }

    pub(crate) fn jit_error_result<R>(
        &self,
        span: &proc_macro2::Span,
        error_message: &str,
    ) -> Result<R, JITError> {
        self.resolve_span(span).jit_error_result(error_message)
    }

    /// Resolve any `proc_macro2::Span` from the current module's syn AST to
    /// an absolute [`SourceLocation`].
    ///
    /// At proc-macro time the module's source text was captured verbatim and
    /// reparsed via `syn::parse_str` at runtime.  Every token span in the
    /// resulting AST therefore has line/column numbers that are offsets within
    /// that string, mapping 1-to-1 to the original file layout.  The
    /// [`SpanBase`] provides the anchor (file, base_line, base_col) to turn
    /// those string-relative positions into absolute ones:
    ///
    /// ```text
    /// abs_line = base_line + (span_line − 1)
    /// abs_col  = if span_line == 1 { base_col + span_col }
    ///            else               { span_col }
    /// ```
    ///
    /// This works for *any* node in the syn AST – statements, expressions,
    /// sub-expressions, individual tokens – without requiring any up-front
    /// walk or key-based lookup table.
    pub(crate) fn resolve_span(&self, span: &proc_macro2::Span) -> SourceLocation {
        self.span_base().resolve_span(span)
    }

    /// Resolve a `proc_macro2::Span` to an MLIR `Location` in one step.
    ///
    /// This is the primary method for attaching source-location information to
    /// MLIR operations.  Given *any* `proc_macro2::Span` from the current
    /// module's syn AST, it produces a `Location::new(ctx, file, line, col)`
    /// that points to the exact position in the user's source code.
    pub(crate) fn location_from_span(&self, span: &proc_macro2::Span) -> Location<'_> {
        self.mlir_location(&self.resolve_span(span))
    }

    /// Return an MLIR `Location` for the function currently being compiled.
    ///
    /// Uses the function item's own span resolved through the module's
    /// [`SpanBase`].  Falls back to `Location::unknown` when no span
    /// information was captured.
    pub(crate) fn function_location(&self) -> Location<'_> {
        self.location_from_span(&self.function.span())
    }

    pub fn get_validator(&self) -> Validator {
        self.validator.clone()
    }

    pub fn gpu_name(&self) -> &str {
        &self.gpu_name
    }

    pub fn compile_function(
        &self,
        fn_item: &ItemFn,
        generic_vars: &GenericVars,
        stride_args: &HashMap<String, Vec<i32>>,
    ) -> Result<Operation<'_>, JITError> {
        let fn_name = fn_item.sig.ident.to_string();
        let var_names = get_sig_param_names(&fn_item.sig);
        let (r_params, r_result) = get_sig_types(&fn_item.sig, None);
        let mut cuda_tile_argument_types = vec![];
        let mut cuda_tile_return_types = vec![];

        for (i, r_param_type) in r_params.iter().enumerate() {
            let mut type_params: HashMap<String, TypeParam> = HashMap::new();
            if let Some(strides) = stride_args.get(var_names[i].as_str()) {
                type_params.insert(
                    "strides".to_string(),
                    TypeParam::Strides(TypeParamStrides::from(
                        syn::parse2::<syn::Type>(
                            format!(
                                "Array<{{[{}]}}>",
                                strides
                                    .iter()
                                    .map(|i| i.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                            .parse()
                            .unwrap(),
                        )
                        .unwrap(),
                    )),
                );
            }
            match self.compile_type(&r_param_type, &generic_vars, &type_params)? {
                Some(ty) => cuda_tile_argument_types.push(ty),
                None => {
                    return self.jit_error_result(
                        &r_param_type.span(),
                        &format!(
                            "unable to compile parameter type `{}`",
                            r_param_type.to_token_stream().to_string()
                        ),
                    );
                }
            }
        }
        match self.compile_type(&r_result, &generic_vars, &HashMap::new())? {
            Some(ty) => cuda_tile_return_types.push(ty),
            None => {}
        }

        let argument_types = cuda_tile_argument_types
            .iter()
            .map(|ct_ty| ct_ty.cuda_tile_ty.clone().unwrap())
            .collect::<Vec<_>>();
        let result_types = cuda_tile_return_types
            .iter()
            .map(|ct_ty| ct_ty.cuda_tile_ty.clone().unwrap())
            .collect::<Vec<_>>();

        let function_type = FunctionType::new(&self.context, &argument_types, &result_types);
        let location = Location::unknown(&self.context);

        let entry_attrs = get_meta_list_by_last_segment("entry", &fn_item.attrs);
        let fn_builder = if entry_attrs.is_some() {
            OperationBuilder::new("cuda_tile.entry", location)
        } else {
            OperationBuilder::new("cuda_tile.func", location)
        };
        let mut attrs = vec![
            named_str_attr(&self.context, "sym_name", fn_name.as_str()),
            named_type_attr(&self.context, "function_type", function_type.into()),
        ];
        if let Some(entry_hints) = self.optimization_hints.get_entry_opt_hints(&self.context)? {
            attrs.push(entry_hints);
        }
        let res = fn_builder
            .add_attributes(&attrs)
            .add_regions([{
                let func_block = Block::new(
                    &argument_types
                        .iter()
                        .map(|&ty| (ty, location))
                        .collect::<Vec<_>>(),
                );
                let sig_param_mutability = get_sig_param_mutability(&fn_item.sig);
                let mut ctx: CompilerContext = CompilerContext::empty();
                for (i, name) in var_names.iter().enumerate() {
                    let ty = cuda_tile_argument_types[i].clone();
                    let value: Value = func_block.argument(i).unwrap().into();
                    let mut val = TileRustValue::new_value_kind_like(value.clone(), ty);
                    val.mutability = if sig_param_mutability[i] {
                        Mutability::Mutable
                    } else {
                        Mutability::Immutable
                    };
                    ctx.vars.insert(name.clone(), val);
                }

                // Add const generics as variables.
                for (key, value) in &generic_vars.inst_i32 {
                    let tr_val = self.compile_constant(&func_block, generic_vars, *value)?;
                    ctx.vars.insert(key.clone(), tr_val);
                }

                // Add arrays as variables.
                for (key, value) in &generic_vars.inst_array {
                    let arr_expr =
                        syn::parse2::<Expr>(format!("{value:?}").parse().unwrap()).unwrap();
                    let arr_ty =
                        syn::parse2::<Type>(format!("[i32;{}]", value.len()).parse().unwrap())
                            .unwrap();
                    let ty = self.compile_type(&arr_ty, generic_vars, &HashMap::new())?;
                    let tr_val = self
                        .compile_expression(&func_block, &arr_expr, generic_vars, &mut ctx, ty)?
                        .expect("Failed to compile CGA as var.");
                    ctx.vars.insert(key.clone(), tr_val);
                }
                ctx.default_terminator = Some(BlockTerminator::Return);
                // TODO (hme): Cannot pass mutable reference to func_block,
                //  but it is still mutable through &?
                //  Refactor to produce func_block.
                let return_value = self.compile_block(
                    &func_block,
                    &*fn_item.block,
                    &generic_vars,
                    &mut ctx,
                    None,
                )?;
                if return_value.is_some() {
                    return self.jit_error_result(
                        &fn_item.block.span(),
                        "returning a value from this function is not supported",
                    );
                }
                let region = Region::new();
                region.append_block(func_block);
                region
            }])
            .build()
            .unwrap()
            .into();
        Ok(res)
    }

    // -----------------------------------------------------------------------
    // Helper / utility methods
    // -----------------------------------------------------------------------

    pub(crate) fn flag_attr(&'c self, name: &str) -> (Identifier<'c>, Attribute<'c>) {
        named_flag_attr(&self.context, name)
    }

    pub(crate) fn parse_named_attr(
        &'c self,
        name: &str,
        attr_string: &str,
    ) -> Result<(Identifier<'c>, Attribute<'c>), crate::error::JITError> {
        parse_named_attr(&self.context, name, attr_string)
    }

    pub fn compile_call_args(
        &'c self,
        builder: &'c Block<'c>,
        args: &Punctuated<syn::Expr, syn::Token![,]>,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
    ) -> Result<Vec<TileRustValue<'c, 'c>>, JITError> {
        let mut result = vec![];
        for arg in args {
            let value = self
                .compile_expression(builder, &arg, generic_args, ctx, None)?
                .ok_or(self.jit_error(
                    &arg.span(),
                    &format!(
                        "Failed to compile argument: {:?}",
                        arg.to_token_stream().to_string()
                    ),
                ))?;
            result.push(value);
        }
        Ok(result)
    }

    // TODO (hme): Get rid of this. It's useful but may emit unused ops.
    // The functions which use this need to be refactored to collect
    // information they need in some other way.
    pub fn compile_call_args_no_side_effect(
        &'c self,
        builder: &'c Block<'c>,
        args: &Punctuated<syn::Expr, syn::Token![,]>,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
    ) -> Result<Vec<TileRustValue<'c, 'c>>, JITError> {
        let mut result = vec![];
        for arg in args {
            let value = self
                .compile_expression(builder, &arg, generic_args, ctx, None)?
                .ok_or(self.jit_error(
                    &arg.span(),
                    &format!(
                        "Failed to compile argument: {:?}",
                        arg.to_token_stream().to_string()
                    ),
                ))?;
            result.push(value);
        }
        Ok(result)
    }

    pub(crate) fn compile_constant<T: Into<i64>>(
        &'c self,
        builder: &'c ir::Block<'c>,
        generic_vars: &GenericVars,
        x: T,
    ) -> Result<TileRustValue<'c, 'c>, JITError> {
        let bounds = Bounds::exact(x.into());
        let rust_ty_str = type_name::<T>();
        let rust_ty = syn::parse2::<syn::Type>(rust_ty_str.parse()?).unwrap();
        let tr_ty = self
            .compile_type(&rust_ty, &generic_vars, &HashMap::new())?
            .ok_or(self.jit_error(&rust_ty.span(), "failed to compile constant"))?;
        self.compile_constant_from_exact_bounds(builder, bounds, tr_ty)
    }

    pub(crate) fn compile_constant_from_exact_bounds(
        &'c self,
        builder: &'c ir::Block<'c>,
        bounds: Bounds<i64>,
        tr_ty: TileRustType<'c>,
    ) -> Result<TileRustValue<'c, 'c>, JITError> {
        if !bounds.is_exact() {
            return self.jit_error_result(
                &tr_ty.rust_ty.span(),
                &format!(
                    "expected a compile-time constant, but got a value with bounds [{}, {}]",
                    bounds.start, bounds.end
                ),
            );
        }
        let const_value = bounds.start;
        let TypeInstance::ElementType(type_inst) = &tr_ty.type_instance else {
            return self.jit_error_result(&tr_ty.rust_ty.span(), "expected a scalar element type");
        };
        let Some(const_ty) = get_cuda_tile_element_type_from_rust_primitive_str(
            &type_inst.rust_element_instance_ty,
            &self.modules.primitives,
        ) else {
            return self
                .jit_error_result(&tr_ty.rust_ty.span(), "failed to compile constant value");
        };
        let mlir_value: Value = builder
            .append_operation(
                operation_parse(
                    &self.context,
                    format!("%0 = cuda_tile.constant <{const_ty}: {const_value}> : !cuda_tile.tile<{const_ty}>").as_str(),
                    None,
                )
                .ok_or(self.jit_error(&tr_ty.rust_ty.span(), "expected a scalar element type"))?,
            )
            .result(0)?
            .into();
        let mut tr_val = TileRustValue::new_value_kind_like(mlir_value, tr_ty);
        tr_val.mutability = Mutability::Immutable;
        tr_val.bounds = Some(bounds);
        Ok(tr_val)
    }
}
