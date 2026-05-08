/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Core compiler struct for compiler2.
//!
//! Self-sufficient compiler that emits tile-ir ops directly, without wrapping
//! the old CUDATileFunctionCompiler.

use super::_module::CUDATileModules;
use crate::ast::{SourceLocation, SpanBase};
use crate::bounds::Bounds;
use crate::error::{JITError, SpannedJITError};
use crate::generics::{GenericVars, TypeInstance};
use crate::kernel_entry_generator::generate_entry_point;
use crate::kernel_naming::KernelNaming;
use crate::syn_utils::*;
use crate::types::get_cuda_tile_element_type_from_rust_primitive_str;
use crate::types::get_sig_param_mutability;
use cuda_async::device_context::Validator;

use super::_value::{BlockTerminator, CompilerContext, Mutability, TileRustValue};
use super::optimization_hints::{build_entry_optimization_hints, OptimizationHints};
use super::shared_types::EntryAttrs;
use super::tile_rust_type::TileRustType;

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{
    Attribute, DenseElements, FuncType, Module, ScalarType, TileElementType, TileType, Type,
};

use anyhow::Context as AnyhowContext;
use quote::ToTokens;
use std::any::type_name;
use std::cell::RefCell;
use std::collections::HashMap;
use syn::spanned::Spanned;

/// Compiles a single Rust function into Tile IR bytecode.
pub struct CUDATileFunctionCompiler<'m> {
    pub(crate) modules: &'m CUDATileModules,
    pub(crate) module_name: String,
    pub(crate) _function_name: String,
    pub(crate) _function: &'m syn::ItemFn,
    pub(crate) entry: syn::ItemFn,
    pub(crate) entry_attrs: EntryAttrs,
    pub(crate) const_grid: Option<(u32, u32, u32)>,
    pub(crate) gpu_name: String,
    pub(crate) optimization_hints: OptimizationHints,
    pub(crate) stride_args: HashMap<String, Vec<i32>>,
    pub(crate) generic_vars: GenericVars,
    pub(crate) validator: Validator,
    pub(crate) module_name_stack: Vec<String>,
    pub(crate) typeck_results: RefCell<Option<crate::passes::type_inference::TypeckResults>>,
}

struct FunctionParamTypes {
    names: Vec<String>,
    tile_types: Vec<TileRustType>,
}

impl<'m> CUDATileFunctionCompiler<'m> {
    pub fn new(
        modules: &'m CUDATileModules,
        module_name: &str,
        function_name: &str,
        function_generic_args: &[String],
        stride_args: &[(&str, &[i32])],
        spec_args: &[(&str, &crate::specialization::SpecializationBits)],
        scalar_hints: &[(&str, &crate::specialization::DivHint)],
        const_grid: Option<(u32, u32, u32)>,
        gpu_name: String,
        compile_options: &crate::hints::CompileOptions,
    ) -> Result<Self, JITError> {
        // 1. Check module exists.
        if !modules.modules().contains_key(module_name) {
            return Err(JITError::Generic(format!(
                "Undefined module: {module_name}"
            )));
        }

        // 2. KernelNaming.
        let kernel_naming = KernelNaming::new(function_name);

        // 3. Look up function.
        let (_, function) = modules
            .functions()
            .get(kernel_naming.public_name())
            .with_context(|| format!("Undefined function: {function_name}"))?;

        // 4. Parse entry_attrs.
        let entry_attrs =
            get_meta_list_by_last_segment("entry", &function.attrs).ok_or_else(|| {
                modules
                    .resolve_span(module_name, &function.span())
                    .jit_error(&format!(
                    "function `{function_name}` is missing a required `#[entry(...)]` attribute"
                ))
            })?;
        let entry_attrs = EntryAttrs { entry_attrs };

        // 5. Check unchecked_accesses.
        if entry_attrs.get_entry_arg_bool("unchecked_accesses") && function.sig.unsafety.is_none() {
            return modules
                .resolve_span(module_name, &function.span())
                .jit_error_result(
                    "kernel must be declared `unsafe` when `unchecked_accesses` is enabled",
                );
        }

        // 6. Parse optimization_hints.
        let mut optimization_hints = match entry_attrs.get_entry_arg_expr("optimization_hints") {
            Some(hints_expr) => OptimizationHints::parse(hints_expr, gpu_name.clone())?,
            None => {
                let mut hints = OptimizationHints::empty();
                hints.target_gpu_name = Some(gpu_name.clone());
                hints
            }
        };
        // Runtime compile options override entry-level hints.
        optimization_hints.apply_compile_options(compile_options);

        // 7. Build stride_args HashMap.
        let stride_args: HashMap<String, Vec<i32>> = stride_args
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_vec()))
            .collect::<HashMap<_, _>>();

        // 8. Create GenericVars.
        let mut generic_vars =
            GenericVars::from_flat(&function.sig.generics, function_generic_args)?;
        Self::add_module_const_vars_from_modules(modules, &mut generic_vars);

        // 9. generate_entry_point.
        let spec_args_map: HashMap<String, crate::specialization::SpecializationBits> = spec_args
            .iter()
            .map(|(k, v)| (k.to_string(), (*v).clone()))
            .collect();
        let scalar_max_divisibility = optimization_hints
            .target_gpu_name
            .as_ref()
            .and_then(|target| optimization_hints.tile_as_hints.get(target))
            .and_then(|hints| hints.max_divisibility);
        let scalar_hints_map: HashMap<String, crate::specialization::DivHint> = scalar_hints
            .iter()
            .map(|&(name, hint)| {
                let hint = scalar_max_divisibility.map_or(*hint, |max| hint.with_max(max));
                (name.to_string(), hint)
            })
            .collect();
        let (entry, validator) = generate_entry_point(
            modules,
            &function,
            &generic_vars,
            &stride_args,
            &spec_args_map,
            &scalar_hints_map,
            &modules.primitives(),
            &optimization_hints,
        )?;

        // 10. Check namespace collision.
        if modules
            .functions()
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

        // 11. Optional print_ir.
        if entry_attrs.get_entry_arg_bool("print_ir") {
            println!("GENERATED ENTRY POINT: {module_name}::{function_name}");
            println!("{}", item_string_pretty(&entry.clone().into()));
            println!();
        }

        // 12. Build struct directly.
        Ok(CUDATileFunctionCompiler {
            modules,
            module_name: module_name.to_string(),
            _function_name: function_name.to_string(),
            entry_attrs,
            const_grid,
            gpu_name,
            optimization_hints,
            _function: function,
            entry,
            validator,
            generic_vars,
            stride_args,
            module_name_stack: vec![module_name.to_string()],
            typeck_results: RefCell::new(None),
        })
    }

    // -----------------------------------------------------------------------
    // Error helper methods
    // -----------------------------------------------------------------------

    pub(crate) fn add_module_const_vars(&self, generic_vars: &mut GenericVars) {
        Self::add_module_const_vars_from_modules(self.modules, generic_vars);
    }

    fn add_module_const_vars_from_modules(
        modules: &CUDATileModules,
        generic_vars: &mut GenericVars,
    ) {
        for (name, item) in modules.consts() {
            if generic_vars.var_type(name).is_some() {
                continue;
            }
            if let Some(value) = crate::type_aliases::const_item_i32_value(item) {
                generic_vars.inst_i32.insert(name.clone(), value);
            } else if let Some(value) = crate::type_aliases::const_item_bool_value(item) {
                generic_vars.inst_bool.insert(name.clone(), value);
            }
        }
    }

    pub(crate) fn span_base(&self) -> SpanBase {
        let current_module = &self.module_name_stack[0];
        self.modules
            .get_span_base(current_module)
            .cloned()
            .unwrap_or_default()
    }

    pub(crate) fn resolve_span(&self, span: &proc_macro2::Span) -> SourceLocation {
        self.span_base().resolve_span(span)
    }

    /// Convert a proc_macro2 span into a tile-ir Location for IR operations.
    pub(crate) fn ir_location(&self, span: &proc_macro2::Span) -> cutile_ir::ir::Location {
        let loc = self.resolve_span(span);
        if loc.is_known() {
            cutile_ir::ir::Location::FileLineCol {
                filename: loc.file,
                line: loc.line as u32,
                column: loc.column as u32,
            }
        } else {
            cutile_ir::ir::Location::Unknown
        }
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

    // -----------------------------------------------------------------------
    // Typeck query helper methods
    // -----------------------------------------------------------------------

    pub(crate) fn typeck_method_selection(
        &self,
        method_call_expr: &syn::ExprMethodCall,
    ) -> Option<crate::passes::type_inference::MethodSelection> {
        self.typeck_results
            .borrow()
            .as_ref()
            .and_then(|results| results.method_selection(method_call_expr).cloned())
    }

    pub(crate) fn typeck_expr_syn_type(&self, expr: &syn::Expr) -> Option<syn::Type> {
        self.typeck_results
            .borrow()
            .as_ref()
            .and_then(|results| results.syn_expr_type(expr))
    }

    pub(crate) fn typeck_expr_tile_type(
        &self,
        expr: &syn::Expr,
        generic_vars: &GenericVars,
        type_params: &HashMap<String, crate::types::TypeParam>,
    ) -> Result<Option<TileRustType>, JITError> {
        let cached_tile_type = self
            .typeck_results
            .borrow()
            .as_ref()
            .and_then(|results| results.expr_type(expr).cloned());
        if cached_tile_type.is_some() {
            return Ok(cached_tile_type);
        }

        let Some(syn_type) = self.typeck_expr_syn_type(expr) else {
            return Ok(None);
        };
        self.compile_type(&syn_type, generic_vars, type_params)
    }

    // -----------------------------------------------------------------------
    // Compile
    // -----------------------------------------------------------------------

    /// Compile the kernel function into a `cutile_ir::Module`.
    pub fn compile(&self) -> Result<Module, JITError> {
        let mut module = Module::new(&self.module_name);
        self.emit_module_globals(&mut module)?;
        let entry_op = self.compile_entry_function(&mut module)?;
        module.functions.push(entry_op);
        Ok(module)
    }

    fn compile_function_param_types(
        &self,
        fn_item: &syn::ItemFn,
        generic_vars: &GenericVars,
    ) -> Result<FunctionParamTypes, JITError> {
        let names = get_sig_param_names(&fn_item.sig);
        let (r_params, _r_result) = get_sig_types(&fn_item.sig, None);
        let mut tile_types = Vec::new();

        for (i, r_param_type) in r_params.iter().enumerate() {
            let mut type_params: HashMap<String, crate::types::TypeParam> = HashMap::new();
            if let Some(strides) = self.stride_args.get(names[i].as_str()) {
                type_params.insert(
                    "strides".to_string(),
                    crate::types::TypeParam::Strides(crate::types::TypeParamStrides::from(
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
            let Some(ty) = self.compile_type(r_param_type, generic_vars, &type_params)? else {
                return self.jit_error_result(
                    &r_param_type.span(),
                    &format!(
                        "unable to compile parameter type `{}`",
                        r_param_type.to_token_stream()
                    ),
                );
            };
            tile_types.push(ty);
        }

        Ok(FunctionParamTypes { names, tile_types })
    }

    fn initial_typeck_types(
        &self,
        param_types: &FunctionParamTypes,
        generic_vars: &GenericVars,
    ) -> Result<HashMap<String, TileRustType>, JITError> {
        let mut initial_types = param_types
            .names
            .iter()
            .cloned()
            .zip(param_types.tile_types.iter().cloned())
            .collect::<HashMap<_, _>>();

        let i32_ty: syn::Type = syn::parse_quote!(i32);
        for key in generic_vars.inst_i32.keys() {
            let Some(ty) = self.compile_type(&i32_ty, generic_vars, &HashMap::new())? else {
                return SourceLocation::unknown()
                    .jit_error_result("unable to compile const generic i32 type");
            };
            initial_types.insert(key.clone(), ty);
        }

        let bool_ty: syn::Type = syn::parse_quote!(bool);
        for key in generic_vars.inst_bool.keys() {
            let Some(ty) = self.compile_type(&bool_ty, generic_vars, &HashMap::new())? else {
                return SourceLocation::unknown()
                    .jit_error_result("unable to compile const generic bool type");
            };
            initial_types.insert(key.clone(), ty);
        }

        for (key, value) in &generic_vars.inst_array {
            let arr_ty =
                syn::parse2::<syn::Type>(format!("[i32;{}]", value.len()).parse().unwrap())
                    .unwrap();
            let Some(ty) = self.compile_type(&arr_ty, generic_vars, &HashMap::new())? else {
                return SourceLocation::unknown()
                    .jit_error_result("unable to compile const generic array type");
            };
            initial_types.insert(key.clone(), ty);
        }

        Ok(initial_types)
    }

    #[doc(hidden)]
    pub fn debug_typeck_dump(&self) -> Result<String, JITError> {
        let fn_item = self._function;
        let generic_vars = &self.generic_vars;
        let param_types = self.compile_function_param_types(fn_item, generic_vars)?;
        let initial_types = self.initial_typeck_types(&param_types, generic_vars)?;

        let mut typed_fn_item = fn_item.clone();
        crate::passes::node_ids::assign_expr_ids(&mut typed_fn_item);
        let typeck_results = crate::passes::type_inference::infer_function(
            self,
            &typed_fn_item,
            generic_vars,
            initial_types,
        )?;
        Ok(typeck_results.debug_dump())
    }

    /// Compile the entry function, returning its OpId.
    fn compile_entry_function(&self, module: &mut Module) -> Result<cutile_ir::ir::OpId, JITError> {
        let fn_item = &self.entry;
        let fn_name = fn_item.sig.ident.to_string();
        let generic_vars = &self.generic_vars;

        let param_types = self.compile_function_param_types(fn_item, generic_vars)?;
        let var_names = &param_types.names;
        let cuda_tile_argument_types = &param_types.tile_types;
        let mut arg_tile_ir_types = Vec::new();
        for ty in cuda_tile_argument_types {
            let tile_ir_ty = super::_type::convert_type(ty).ok_or_else(|| {
                JITError::Generic(format!(
                    "compiler2: failed to convert parameter type to tile-ir: {}",
                    ty.rust_ty.to_token_stream()
                ))
            })?;
            arg_tile_ir_types.push(tile_ir_ty);
        }

        let func_type = Type::Func(FuncType {
            inputs: arg_tile_ir_types.clone(),
            results: vec![],
        });

        let (region_id, block_id, block_args) =
            build_single_block_region(module, &arg_tile_ir_types);

        // Bind parameter names to block argument values using ported CompilerContext.
        let sig_param_mutability = get_sig_param_mutability(&fn_item.sig);
        let mut ctx = CompilerContext::empty();
        for (i, name) in var_names.iter().enumerate() {
            if i < block_args.len() {
                let ty = cuda_tile_argument_types[i].clone();
                let mut val = TileRustValue::new_value_kind_like(block_args[i], ty);
                val.mutability = if sig_param_mutability[i] {
                    Mutability::Mutable
                } else {
                    Mutability::Immutable
                };
                ctx.vars.insert(name.clone(), val);
            }
        }

        let initial_types = self.initial_typeck_types(&param_types, generic_vars)?;

        // Add const generics as variables.
        for (key, value) in &generic_vars.inst_i32 {
            let tr_val = self.compile_constant(module, block_id, generic_vars, *value)?;
            ctx.vars.insert(key.clone(), tr_val);
        }

        for (key, value) in &generic_vars.inst_bool {
            let tr_val = self.compile_bool_constant(module, block_id, generic_vars, *value)?;
            ctx.vars.insert(key.clone(), tr_val);
        }

        // Add arrays as variables.
        for (key, value) in &generic_vars.inst_array {
            let arr_expr = syn::parse2::<syn::Expr>(format!("{value:?}").parse().unwrap()).unwrap();
            let arr_ty =
                syn::parse2::<syn::Type>(format!("[i32;{}]", value.len()).parse().unwrap())
                    .unwrap();
            let ty = self.compile_type(&arr_ty, generic_vars, &HashMap::new())?;
            let tr_val = self
                .compile_expression(module, block_id, &arr_expr, generic_vars, &mut ctx, ty)?
                .expect("Failed to compile CGA as var.");
            ctx.vars.insert(key.clone(), tr_val);
        }

        ctx.default_terminator = Some(BlockTerminator::Return);

        let mut typed_fn_item = fn_item.clone();
        crate::passes::node_ids::assign_expr_ids(&mut typed_fn_item);
        let typeck_results = crate::passes::type_inference::infer_function(
            self,
            &typed_fn_item,
            generic_vars,
            initial_types,
        )?;
        let lowered_fn_item =
            crate::passes::typed_dispatch_lowering::lower_function(&typed_fn_item, &typeck_results);
        let previous_typeck_results = self.typeck_results.replace(Some(typeck_results));

        if std::env::var("CUTILE_DEBUG_COMPILER2").is_ok() {
            eprintln!(
                "compiler2: lowered entry function body:\n{}",
                quote::quote!(#lowered_fn_item).to_string()
            );
        }

        let return_value = self.compile_block(
            module,
            block_id,
            &*lowered_fn_item.block,
            generic_vars,
            &mut ctx,
            None,
        );
        self.typeck_results.replace(previous_typeck_results);
        let return_value = return_value?;
        if return_value.is_some() {
            return self.jit_error_result(
                &fn_item.block.span(),
                "returning a value from this function is not supported",
            );
        }

        let entry_location = self.ir_location(&fn_item.sig.ident.span());
        let mut entry_builder = OpBuilder::new(Opcode::Entry, entry_location)
            .attr("sym_name", Attribute::String(fn_name))
            .attr("function_type", Attribute::Type(func_type))
            .region(region_id);

        // Forward optimization hints from the parsed hints.
        if let Some(hints_attr) = build_entry_optimization_hints(&self.optimization_hints) {
            entry_builder = entry_builder.attr("optimization_hints", hints_attr);
        }

        let (entry_id, _) = entry_builder.build(module);

        Ok(entry_id)
    }

    pub fn get_validator(&self) -> Validator {
        self.validator.clone()
    }

    pub fn gpu_name(&self) -> &str {
        &self.gpu_name
    }

    // -----------------------------------------------------------------------
    // Helper methods ported from _function.rs
    // -----------------------------------------------------------------------

    pub fn compile_call_args(
        &self,
        module: &mut Module,
        block_id: cutile_ir::ir::BlockId,
        args: &syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
    ) -> Result<Vec<TileRustValue>, JITError> {
        let mut result = vec![];
        for arg in args {
            let expected = if matches!(arg, syn::Expr::Lit(_) | syn::Expr::Unary(_)) {
                self.typeck_expr_tile_type(arg, generic_args, &HashMap::new())?
            } else {
                None
            };
            let value = self
                .compile_expression(module, block_id, &arg, generic_args, ctx, expected)?
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

    pub fn compile_call_args_no_side_effect(
        &self,
        module: &mut Module,
        block_id: cutile_ir::ir::BlockId,
        args: &syn::punctuated::Punctuated<syn::Expr, syn::Token![,]>,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
    ) -> Result<Vec<TileRustValue>, JITError> {
        self.compile_call_args(module, block_id, args, generic_args, ctx)
    }

    pub(crate) fn compile_constant<T: Into<i64>>(
        &self,
        module: &mut Module,
        block_id: cutile_ir::ir::BlockId,
        generic_vars: &GenericVars,
        x: T,
    ) -> Result<TileRustValue, JITError> {
        let bounds = Bounds::exact(x.into());
        let rust_ty_str = type_name::<T>();
        let rust_ty = syn::parse2::<syn::Type>(rust_ty_str.parse()?).unwrap();
        let tr_ty = self
            .compile_type(&rust_ty, &generic_vars, &HashMap::new())?
            .ok_or(self.jit_error(&rust_ty.span(), "failed to compile constant"))?;
        self.compile_constant_from_exact_bounds(module, block_id, bounds, tr_ty)
    }

    pub(crate) fn compile_bool_constant(
        &self,
        module: &mut Module,
        block_id: cutile_ir::ir::BlockId,
        generic_vars: &GenericVars,
        x: bool,
    ) -> Result<TileRustValue, JITError> {
        let rust_ty: syn::Type = syn::parse_quote!(bool);
        let tr_ty = self
            .compile_type(&rust_ty, generic_vars, &HashMap::new())?
            .ok_or(self.jit_error(&rust_ty.span(), "failed to compile bool constant"))?;
        self.compile_constant_from_exact_bounds(
            module,
            block_id,
            Bounds::exact(if x { 1 } else { 0 }),
            tr_ty,
        )
    }

    pub(crate) fn compile_constant_from_exact_bounds(
        &self,
        module: &mut Module,
        block_id: cutile_ir::ir::BlockId,
        bounds: Bounds<i64>,
        tr_ty: TileRustType,
    ) -> Result<TileRustValue, JITError> {
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
        let Some(const_ty_str) = get_cuda_tile_element_type_from_rust_primitive_str(
            &type_inst.rust_element_instance_ty,
            &self.modules.primitives(),
        ) else {
            return self
                .jit_error_result(&tr_ty.rust_ty.span(), "failed to compile constant value");
        };

        // Build tile-ir Constant op directly (replaces operation_parse).
        let scalar = super::_type::scalar_from_name(&const_ty_str).ok_or_else(|| {
            JITError::Generic(format!(
                "unsupported scalar type for constant: {const_ty_str}"
            ))
        })?;
        let result_ty = Type::Tile(TileType {
            shape: vec![],
            element_type: TileElementType::Scalar(scalar),
        });
        let data = match scalar {
            ScalarType::I1 => vec![if const_value != 0 { 0xFF } else { 0x00 }],
            ScalarType::I8 => (const_value as i8).to_le_bytes().to_vec(),
            ScalarType::I16 => (const_value as i16).to_le_bytes().to_vec(),
            ScalarType::I32 => (const_value as i32).to_le_bytes().to_vec(),
            ScalarType::I64 => const_value.to_le_bytes().to_vec(),
            ScalarType::F16 => half::f16::from_f64(const_value as f64)
                .to_le_bytes()
                .to_vec(),
            ScalarType::BF16 => half::bf16::from_f64(const_value as f64)
                .to_le_bytes()
                .to_vec(),
            ScalarType::F32 => (const_value as f32).to_le_bytes().to_vec(),
            ScalarType::F64 => (const_value as f64).to_le_bytes().to_vec(),
            _ => (const_value as i32).to_le_bytes().to_vec(),
        };
        let (op_id, results) =
            OpBuilder::new(Opcode::Constant, self.ir_location(&tr_ty.rust_ty.span()))
                .result(result_ty.clone())
                .attr(
                    "value",
                    Attribute::DenseElements(DenseElements {
                        element_type: result_ty,
                        shape: vec![],
                        data,
                    }),
                )
                .build(module);
        append_op(module, block_id, op_id);
        let mut tr_val = TileRustValue::new_value_kind_like(results[0], tr_ty);
        tr_val.mutability = Mutability::Immutable;
        tr_val.bounds = Some(bounds);
        Ok(tr_val)
    }

    /// Return the typeck side-table type for an expression when possible, then
    /// fall back to op-signature derivation for types whose Tile IR params are
    /// completed from call arguments.
    ///
    /// The old recursive type derivation path is still available behind
    /// `CUTILE_DERIVE_TYPE_LEGACY` for debugging, but the compiler should
    /// normally consume Pass 2 typeck results instead of re-deriving types.
    pub(crate) fn derive_type(
        &self,
        module: &mut Module,
        block_id: cutile_ir::ir::BlockId,
        expr: &syn::Expr,
        maybe_type_params: Option<Vec<crate::types::TypeParam>>,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
    ) -> Result<Option<TileRustType>, JITError> {
        use crate::generics::GenericArgInference;
        use crate::types::TypeParam;
        use syn::Expr;

        if std::env::var_os("CUTILE_DERIVE_TYPE_LEGACY").is_none() {
            let typeck_type_params = maybe_type_params
                .as_ref()
                .map(|type_params| {
                    type_params
                        .iter()
                        .filter_map(|type_param| {
                            type_param
                                .name()
                                .map(|name| (name.to_string(), type_param.clone()))
                        })
                        .collect::<HashMap<_, _>>()
                })
                .unwrap_or_default();
            if let Some(return_type) =
                self.typeck_expr_tile_type(expr, generic_vars, &typeck_type_params)?
            {
                return Ok(Some(return_type));
            }
        }

        match expr {
            Expr::MethodCall(method_call_expr) => {
                let ident = &method_call_expr.method;
                if let Some(return_type) = self
                    .typeck_method_selection(method_call_expr)
                    .and_then(|selection| selection.return_type)
                {
                    return Ok(Some(return_type));
                }
                let mut args = method_call_expr.args.clone();
                args.insert(0, *method_call_expr.receiver.clone());
                let call_arg_values = self.compile_call_args_no_side_effect(
                    module,
                    block_id,
                    &args,
                    generic_vars,
                    ctx,
                )?;
                let call_arg_rust_tys = call_arg_values
                    .iter()
                    .map(|arg| arg.ty.rust_ty.clone())
                    .collect::<Vec<_>>();
                let receiver_rust_ty = &call_arg_rust_tys[0];
                let Some((_, impl_item, impl_method)) = self.modules.get_impl_item_fn(
                    receiver_rust_ty,
                    method_call_expr,
                    generic_vars,
                    &call_arg_rust_tys,
                )?
                else {
                    return self.jit_error_result(
                        &method_call_expr.method.span(),
                        &format!("Undefined method {ident}"),
                    );
                };
                let self_ty = &*impl_item.self_ty;
                let (fn_arg_types, return_type) = get_sig_types(&impl_method.sig, Some(self_ty));

                if call_arg_values.len() != fn_arg_types.len() {
                    return self.jit_error_result(
                        &method_call_expr.method.span(),
                        &format!(
                            "Argument count mismatch for method {}: expected {} args, got {} compiled values",
                            method_call_expr.method.to_string(),
                            fn_arg_types.len(),
                            call_arg_values.len()
                        ),
                    );
                }

                let mut arg_types: HashMap<String, TileRustType> = HashMap::new();
                let mut arg_string_values: HashMap<String, String> = HashMap::new();
                let mut arg_zst_values: HashMap<String, String> = HashMap::new();
                for (i, param_name) in get_sig_param_names(&impl_method.sig).iter().enumerate() {
                    if i < call_arg_values.len() {
                        let call_arg_val = &call_arg_values[i];
                        let call_arg_ty = call_arg_val.ty.clone();
                        if let Some(ref string_lit_expr) = call_arg_val.string_literal {
                            if let Some(value) = super::shared_utils::zst_type_name(string_lit_expr)
                            {
                                arg_zst_values.insert(param_name.to_string(), value);
                            }
                            if let Expr::Lit(lit_expr) = string_lit_expr {
                                if let syn::Lit::Str(s) = &lit_expr.lit {
                                    arg_string_values.insert(param_name.to_string(), s.value());
                                }
                            } else if param_name == "padding_value" {
                                if let Some(value) =
                                    super::shared_utils::padding_zst_value(string_lit_expr)
                                {
                                    arg_string_values.insert(param_name.to_string(), value);
                                }
                            }
                        }
                        arg_types.insert(param_name.to_string(), call_arg_ty);
                    }
                }

                let mut generic_arg_inf = GenericArgInference::new_method(&impl_item, &impl_method);
                generic_arg_inf.map_args_to_params(&call_arg_rust_tys, Some(self_ty));
                generic_arg_inf
                    .apply_provided_generics_method_call(&method_call_expr, generic_vars);
                if !generic_arg_inf.verify() {
                    return self.jit_error_result(
                        &method_call_expr.method.span(),
                        &format!(
                            "Failed to infer all generic parameters for {}",
                            user_method_call_tokens(method_call_expr)
                        ),
                    );
                }

                let call_output_type: syn::Type =
                    generic_arg_inf.infer_type(&return_type, generic_vars);
                let mut type_params: HashMap<String, TypeParam> = HashMap::new();
                if let Some(given_type_params) = maybe_type_params {
                    for type_param in given_type_params {
                        if let Some(name) = type_param.name() {
                            type_params.insert(name.to_string(), type_param.clone());
                        } else {
                            return self.jit_error_result(
                                &method_call_expr.method.span(),
                                &format!("Failed to get name for type param {type_param:?}"),
                            );
                        }
                    }
                }
                if let Some(op_attrs) = self
                    .modules
                    .get_cuda_tile_op_attrs(ident.to_string().as_str())
                {
                    if let Some(output_type_params) =
                        op_attrs.parse_string_arr("output_type_params")
                    {
                        for type_param_name in output_type_params {
                            if should_skip_optional_output_type_param(
                                &type_param_name,
                                &arg_zst_values,
                            ) {
                                continue;
                            }
                            match arg_types.get(&type_param_name) {
                                Some(arg_type) => {
                                    let cuda_tile_type_str = arg_type.get_cuda_tile_type_str();
                                    let type_instance = Some(arg_type.type_instance.clone());
                                    let mut type_param = TypeParam::derive_param_from_type(
                                        type_param_name.clone(),
                                        arg_type.rust_ty.clone(),
                                        cuda_tile_type_str,
                                        type_instance,
                                    );
                                    if let TypeParam::Padding(ref mut padding) = type_param {
                                        padding.padding_value =
                                            arg_string_values.get(&type_param_name).cloned();
                                    }
                                    type_params.insert(type_param_name.to_string(), type_param);
                                }
                                None => {
                                    return self.jit_error_result(
                                        &method_call_expr.method.span(),
                                        &format!("Unable to find output type: {type_param_name}"),
                                    )
                                }
                            }
                        }
                    }
                }
                let ct_type = self.compile_type(&call_output_type, generic_vars, &type_params)?;
                if ct_type.is_none() {
                    return self.jit_error_result(
                        &method_call_expr.method.span(),
                        &format!(
                            "Failed to derive output for {} \ncall_output_type={}",
                            user_method_call_tokens(method_call_expr),
                            call_output_type.to_token_stream().to_string()
                        ),
                    );
                }
                Ok(ct_type)
            }
            Expr::Call(call_expr) => match &*call_expr.func {
                Expr::Path(path_expr) => {
                    let ident = get_ident_from_path_expr(&path_expr);
                    let Some((_, fn_item)) = self.modules.get_function_by_name(&ident.to_string())
                    else {
                        return self.jit_error_result(
                            &call_expr.func.span(),
                            &format!("Undefined function {ident}"),
                        );
                    };
                    let call_arg_values = self.compile_call_args_no_side_effect(
                        module,
                        block_id,
                        &call_expr.args,
                        generic_vars,
                        ctx,
                    )?;
                    let (fn_arg_types, return_type) = get_sig_types(&fn_item.sig, None);

                    if call_arg_values.len() != fn_arg_types.len() {
                        return self.jit_error_result(
                                &call_expr.func.span(),
                                &format!(
                                    "Argument count mismatch for {}: expected {} args, got {} compiled values",
                                    ident.to_string(),
                                    fn_arg_types.len(),
                                    call_arg_values.len()
                                ),
                            );
                    }

                    let mut call_arg_rust_tys = vec![];
                    let mut arg_types: HashMap<String, TileRustType> = HashMap::new();
                    let mut arg_string_values: HashMap<String, String> = HashMap::new();
                    let mut arg_zst_values: HashMap<String, String> = HashMap::new();
                    for (i, param_name) in get_sig_param_names(&fn_item.sig).iter().enumerate() {
                        if i < call_arg_values.len() {
                            let call_arg_val = &call_arg_values[i];
                            let call_arg_ty = call_arg_val.ty.clone();
                            call_arg_rust_tys.push(call_arg_ty.rust_ty.clone());
                            if let Some(ref string_lit_expr) = call_arg_val.string_literal {
                                if let Some(value) =
                                    super::shared_utils::zst_type_name(string_lit_expr)
                                {
                                    arg_zst_values.insert(param_name.to_string(), value);
                                }
                                if let Expr::Lit(lit_expr) = string_lit_expr {
                                    if let syn::Lit::Str(s) = &lit_expr.lit {
                                        arg_string_values.insert(param_name.to_string(), s.value());
                                    }
                                } else if param_name == "padding_value" {
                                    if let Some(value) =
                                        super::shared_utils::padding_zst_value(string_lit_expr)
                                    {
                                        arg_string_values.insert(param_name.to_string(), value);
                                    }
                                }
                            }
                            arg_types.insert(param_name.to_string(), call_arg_ty);
                        }
                    }

                    let mut generic_arg_inf =
                        GenericArgInference::new_function(fn_item.sig.clone());
                    generic_arg_inf.map_args_to_params(&call_arg_rust_tys, None);
                    generic_arg_inf.apply_provided_generics_fn_call(&call_expr, generic_vars);
                    if !generic_arg_inf.verify() {
                        return self.jit_error_result(
                            &call_expr.func.span(),
                            &format!(
                                "Failed to infer all generic parameters for {}",
                                user_call_tokens(call_expr)
                            ),
                        );
                    }

                    let call_output_type: syn::Type =
                        generic_arg_inf.infer_type(&return_type, generic_vars);
                    let mut type_params: HashMap<String, TypeParam> = HashMap::new();
                    if let Some(given_type_params) = maybe_type_params {
                        for type_param in given_type_params {
                            if let Some(name) = type_param.name() {
                                type_params.insert(name.to_string(), type_param.clone());
                            } else {
                                return self.jit_error_result(
                                    &call_expr.func.span(),
                                    &format!("Failed to get name for type param {type_param:?}"),
                                );
                            }
                        }
                    }
                    if let Some(op_attrs) = self
                        .modules
                        .get_cuda_tile_op_attrs(ident.to_string().as_str())
                    {
                        if let Some(output_type_params) =
                            op_attrs.parse_string_arr("output_type_params")
                        {
                            for type_param_name in output_type_params {
                                if should_skip_optional_output_type_param(
                                    &type_param_name,
                                    &arg_zst_values,
                                ) {
                                    continue;
                                }
                                match arg_types.get(&type_param_name) {
                                    Some(arg_type) => {
                                        let cuda_tile_type_str = arg_type.get_cuda_tile_type_str();
                                        let mut type_param = TypeParam::derive_param_from_type(
                                            type_param_name.clone(),
                                            arg_type.rust_ty.clone(),
                                            cuda_tile_type_str,
                                            Some(arg_type.type_instance.clone()),
                                        );
                                        if let TypeParam::Padding(ref mut padding) = type_param {
                                            padding.padding_value =
                                                arg_string_values.get(&type_param_name).cloned();
                                        }
                                        type_params.insert(type_param_name.to_string(), type_param);
                                    }
                                    None => {
                                        return self.jit_error_result(
                                            &call_expr.func.span(),
                                            &format!(
                                                "Unable to find output type: {type_param_name}"
                                            ),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    let ct_type =
                        self.compile_type(&call_output_type, generic_vars, &type_params)?;
                    if ct_type.is_none() {
                        return self.jit_error_result(
                                &call_expr.func.span(),
                                &format!(
                                    "Failed to derive output for {} \ngeneric_vars={generic_vars:#?} \ntype_params={type_params:#?}",
                                    user_call_tokens(call_expr)
                                ),
                            );
                    }
                    Ok(ct_type)
                }
                Expr::Closure(_) => {
                    return self.jit_error_result(
                            &call_expr.func.span(),
                            &format!(
                                "Closure calls are not supported.\n\
                                 Closures can only be used as arguments to operations like reduce() or scan().\n\
                                 Found: {}",
                                user_call_tokens(call_expr)
                            ),
                        );
                }
                _ => {
                    return self.jit_error_result(
                        &call_expr.func.span(),
                        &format!(
                            "Type derivation for {} not supported.",
                            call_expr.func.to_token_stream().to_string()
                        ),
                    )
                }
            },
            Expr::Field(field_expr) => {
                let Some(base) = self.compile_expression(
                    module,
                    block_id,
                    &field_expr.base,
                    generic_vars,
                    ctx,
                    None,
                )?
                else {
                    return self.jit_error_result(
                        &field_expr.base.span(),
                        &format!(
                            "Failed to compile {}",
                            field_expr.to_token_stream().to_string()
                        ),
                    );
                };
                let syn::Member::Named(field_name) = &field_expr.member else {
                    return self.jit_error_result(
                        &field_expr.member.span(),
                        "Only named member accesses are supported.",
                    );
                };
                if !base.fields.is_some() {
                    return self.jit_error_result(
                        &field_expr.base.span(),
                        &format!("Expected struct value, found: {base:#?}"),
                    );
                }
                let fields = &base.fields.clone().unwrap();
                let Some(field_value) = fields.get(&field_name.to_string()) else {
                    return self.jit_error_result(
                        &field_expr.member.span(),
                        &format!("{} is not a field in {base:#?}.", field_name.to_string()),
                    );
                };
                Ok(Some(field_value.ty.clone()))
            }
            _ => Ok(None),
        }
    }
}

fn should_skip_optional_output_type_param(
    type_param_name: &str,
    arg_zst_values: &HashMap<String, String>,
) -> bool {
    matches!(
        (
            type_param_name,
            arg_zst_values.get(type_param_name).map(String::as_str)
        ),
        ("padding_value", Some("None")) | ("dim_map", Some("Identity"))
    )
}
