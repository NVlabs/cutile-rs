/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type compilation: translates Rust `syn::Type` AST nodes into `TileRustType` representations
//! used by the CUDA Tile compiler.

pub use crate::compiler::_type::*;
pub use crate::compiler::_value::*;

use crate::compiler::CUDATileFunctionCompiler;
use crate::error::JITError;
use crate::generics::{
    GenericArgInference, GenericVars, Instantiable, TypeInstance, TypeInstanceUserType,
};
use crate::syn_utils::*;
use crate::types::*;
use melior::ir;
use quote::ToTokens;
use std::collections::HashMap;
use syn::spanned::Spanned;
use syn::{Expr, ItemImpl, ItemStruct, Type};

impl<'m, 'c> CUDATileFunctionCompiler<'m> {
    pub fn compile_type(
        &'c self,
        ty: &syn::Type,
        generic_vars: &GenericVars,
        type_params: &HashMap<String, TypeParam>,
    ) -> Result<Option<TileRustType<'c>>, JITError> {
        let _ty_debug_str = ty.to_token_stream().to_string();
        let mut ty_attrs: Option<SingleMetaList> = None;
        let mut structure: Option<(String, &ItemStruct)> = None;
        let mut _maybe_generic_element_type: Option<(String, &ItemImpl)> = None;
        let type_instance: Option<TypeInstance>;

        match ty {
            // Array, Slice, and Tuple compile to the same compiler representation (compound values).
            Type::Tuple(tuple) => {
                if tuple.elems.len() == 0 {
                    return Ok(None);
                } else {
                    let unknown_type_instance = TypeInstanceUserType::instantiate(
                        &ty,
                        generic_vars,
                        &self.modules.primitives,
                    )
                    .unwrap();
                    let type_instance = TypeInstance::UserType(unknown_type_instance);
                    return Ok(Some(TileRustType::new_compound(
                        &self.context,
                        type_instance,
                    )));
                }
            }
            Type::Array(_) => {
                // Arrays in function parameters should be treated as compound unknown types
                // The actual element handling happens in compile_expression for array literals
                let unknown_type_instance =
                    TypeInstanceUserType::instantiate(&ty, generic_vars, &self.modules.primitives)
                        .unwrap();
                let type_instance = TypeInstance::UserType(unknown_type_instance);
                return Ok(Some(TileRustType::new_compound(
                    &self.context,
                    type_instance,
                )));
            }
            Type::Slice(_) => {
                // Slices in function parameters should be treated as compound unknown types
                // The actual element handling happens in compile_expression for array literals
                let unknown_type_instance =
                    TypeInstanceUserType::instantiate(&ty, generic_vars, &self.modules.primitives)
                        .unwrap();
                let type_instance = TypeInstance::UserType(unknown_type_instance);
                return Ok(Some(TileRustType::new_compound(
                    &self.context,
                    type_instance,
                )));
            }
            Type::Reference(ref_ty) => {
                // This is okay since rust_ty is always the provided type we're compiling.
                let mut res = self.compile_type(&*ref_ty.elem, generic_vars, type_params)?;
                match &mut res {
                    Some(cuda_tile_ty) => {
                        cuda_tile_ty.rust_ty = ty.clone();
                    }
                    None => return self.jit_error_result(&ty.span(), "Failed to compile type"),
                }
                return Ok(res);
            }
            Type::Path(_) => {
                match get_type_ident(ty) {
                    Some(ident) => {
                        // This could be a user defined struct, a structured type, or primitive.
                        let type_name = ident.to_string();
                        if let Some(item_struct) = self.modules.structs.get(type_name.as_str()) {
                            ty_attrs = self.modules.get_cuda_tile_type_attrs(type_name.as_str());
                            if ty_attrs.is_none() {
                                // This is a user-defined (not a cuda_tile) struct.
                                // There is no corresponding cuda_tile type.
                                let unknown_type_instance = TypeInstanceUserType::instantiate(
                                    &ty,
                                    generic_vars,
                                    &self.modules.primitives,
                                )
                                .unwrap();
                                let type_instance = TypeInstance::UserType(unknown_type_instance);
                                return Ok(Some(TileRustType::new_structure(
                                    &self.context,
                                    type_name.clone(),
                                    type_instance,
                                )));
                            }
                            structure = Some((type_name.clone(), &item_struct));
                            type_instance =
                                Some(generic_vars.instantiate_type(ty, &self.modules.primitives)?);
                        } else {
                            let local_type_instance =
                                generic_vars.instantiate_type(ty, &self.modules.primitives)?;
                            if let TypeInstance::StringType(_string_inst) = local_type_instance {
                                return Ok(Some(TileRustType::new_string(
                                    &self.context,
                                    TypeInstance::StringType(_string_inst),
                                )));
                            }
                            let Some(element_type_instance_str) =
                                local_type_instance.get_rust_element_instance_ty()
                            else {
                                return self.jit_error_result(&ty.span(), "Failed to compile type");
                            };
                            if let Some(element_type_impl) = self.modules.primitives.get(&(
                                "ElementType".to_string(),
                                element_type_instance_str.to_string(),
                            )) {
                                ty_attrs = self.modules.get_primitives_attrs(
                                    "ElementType",
                                    &element_type_instance_str,
                                );
                                _maybe_generic_element_type =
                                    Some((type_name.clone(), element_type_impl));
                            }
                            if ty_attrs.is_none() {
                                return self.jit_error_result(&ty.span(), "Failed to compile type");
                            }
                            type_instance = Some(local_type_instance);
                        }
                    }
                    None => return self.jit_error_result(&ty.span(), "Failed to compile type"),
                }
            }
            Type::Ptr(_) => {
                let type_name = get_type_ident(&ty);
                if type_name.is_none() {
                    return self.jit_error_result(&ty.span(), "Failed to compile type");
                }
                let type_name = type_name.unwrap().to_string();
                let local_type_instance =
                    generic_vars.instantiate_type(ty, &self.modules.primitives)?;
                let Some(element_type_instance_str) =
                    local_type_instance.get_rust_element_instance_ty()
                else {
                    return self.jit_error_result(&ty.span(), "Failed to compile type");
                };
                if let Some(element_type_impl) = self.modules.primitives.get(&(
                    "ElementType".to_string(),
                    element_type_instance_str.to_string(),
                )) {
                    ty_attrs = self
                        .modules
                        .get_primitives_attrs("ElementType", &element_type_instance_str);
                    _maybe_generic_element_type = Some((type_name.clone(), element_type_impl));
                }
                if ty_attrs.is_none() {
                    return self.jit_error_result(&ty.span(), "Failed to compile type");
                }
                type_instance = Some(local_type_instance);
            }
            _ => return self.jit_error_result(&ty.span(), "Failed to compile type"),
        };

        // Compile cuda_tile type.
        if ty_attrs.is_none() {
            return self.jit_error_result(&ty.span(), "Unexpected type");
        }
        let ty_attrs = ty_attrs.unwrap();
        if structure.is_some() {
            let _ty_str = ty.to_token_stream().to_string();
            // Expecting a structured cuda_tile type.
            // This can be a struct with named fields or index fields (tuple).
            let cuda_tile_attr_name_vec = ty_attrs.name_as_vec().unwrap();
            let attr_name = *cuda_tile_attr_name_vec.last().unwrap();
            if attr_name != "ty" {
                return self
                    .jit_error_result(&ty.span(), &format!("Unexpected attribute: {ty_attrs:#?}"));
            }
            let type_name = ty_attrs.parse_string("name");
            if type_name.is_none() {
                return self.jit_error_result(
                    &ty.span(),
                    &format!(
                        "Unable to compile compiling type {} using attrs {ty_attrs:#?}",
                        ty.to_token_stream().to_string()
                    ),
                );
            }
            let type_name = type_name.unwrap();
            let params = ty_attrs.parse_string_arr("type_params").unwrap_or(vec![]);
            let mut args: Vec<TypeParam> = vec![];
            for param in &params {
                if let Some(type_param) = type_params.get(param) {
                    args.push(type_param.clone());
                    continue;
                }
                match param.as_str() {
                    "{D}xE" => args.push(TypeParam::derive_param_from_type(
                        param.clone(),
                        ty.clone(),
                        None,
                        type_instance.clone(),
                    )),
                    "{D}xP" => args.push(TypeParam::derive_param_from_type(
                        param.clone(),
                        ty.clone(),
                        None,
                        type_instance.clone(),
                    )),
                    "tile" => args.push(TypeParam::derive_param_from_type(
                        param.clone(),
                        ty.clone(),
                        None,
                        type_instance.clone(),
                    )),
                    // In these cases, we need to infer the type.
                    // This *should* only happen for type constructors
                    // for types with static type parameters without corresponding
                    // generic parameters.
                    "strides" => return Ok(None),
                    "tensor_view" => return Ok(None),
                    _ => {
                        return self.jit_error_result(
                            &ty.span(),
                            &format!(
                                "Unexpected param {param} for {:?}",
                                ty.to_token_stream().to_string()
                            ),
                        );
                    }
                }
            }
            let type_params_optional = ty_attrs
                .parse_string_arr("type_params_optional")
                .unwrap_or(vec![]);
            for optional_type_param in &type_params_optional {
                if let Some(type_param) = type_params.get(optional_type_param) {
                    args.push(type_param.clone());
                    continue;
                }
            }
            let type_instance = type_instance.expect("Failed to instantiate type.");
            return Ok(Some(TileRustType::new_structured_type(
                &self.context,
                type_name,
                generic_vars,
                &self.modules.primitives,
                args,
                type_instance,
            )?));
        }
        // Expecting a primitive cuda_tile type (either E or *mut E).
        // "E" means it's implemented for all element types.
        if let Some(TypeInstance::ElementType(element_instance)) = type_instance {
            let Some(scalar_attrs) = self.modules.get_primitives_attrs("Scalar", "E") else {
                return self
                    .jit_error_result(&ty.span(), "misconfigured Scalar impl in core module");
            };
            let type_name = scalar_attrs.parse_string("name");
            if type_name.is_none() {
                return self.jit_error_result(
                    &ty.span(),
                    &format!(
                        "Unable to compile type {} using attrs {scalar_attrs:#?}",
                        element_instance.generic_ty.to_token_stream().to_string()
                    ),
                );
            }
            let type_name = type_name.unwrap();
            // "E" is the value given by element_type.
            let args: Vec<TypeParam> = vec![TypeParam::derive_param_from_type(
                "E".to_string(),
                element_instance.generic_ty.clone(),
                None,
                Some(TypeInstance::ElementType(element_instance.clone())),
            )];
            return Ok(Some(TileRustType::new_primitive_type(
                &self.context,
                type_name,
                generic_vars,
                &self.modules.primitives,
                args,
                TypeInstance::ElementType(element_instance),
            )?));
        } else if let Some(TypeInstance::PtrType(ptr_instance)) = type_instance {
            let Some(pointer_attrs) = self.modules.get_primitives_attrs("Pointer", "* mut E")
            else {
                return self
                    .jit_error_result(&ty.span(), "misconfigured Pointer impl in core module");
            };
            let type_name = pointer_attrs.parse_string("name");
            if type_name.is_none() {
                return self.jit_error_result(
                    &ty.span(),
                    &format!(
                        "Unable to compile compiling type {} using attrs {pointer_attrs:#?}",
                        ty.to_token_stream().to_string()
                    ),
                );
            }
            let type_name = type_name.unwrap();
            // "E" is the value given by element_type.
            let args: Vec<TypeParam> = vec![TypeParam::derive_param_from_type(
                "!cuda_tile.ptr<E>".to_string(),
                ptr_instance.generic_ty.clone(),
                None,
                Some(TypeInstance::PtrType(ptr_instance.clone())),
            )];
            return Ok(Some(TileRustType::new_primitive_type(
                &self.context,
                type_name,
                generic_vars,
                &self.modules.primitives,
                args,
                TypeInstance::PtrType(ptr_instance),
            )?));
        } else {
            return self.jit_error_result(
                &ty.span(),
                &format!("Unable to instantiate Scalar or Pointer impls: type_instance={type_instance:#?}"),
            );
        }
    }
    pub(crate) fn derive_type(
        &'c self,
        builder: &'c ir::Block<'c>,
        expr: &syn::Expr,
        maybe_type_params: Option<Vec<TypeParam>>,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
    ) -> Result<Option<TileRustType<'c>>, JITError> {
        match expr {
            Expr::MethodCall(method_call_expr) => {
                let ident = &method_call_expr.method;
                let mut args = method_call_expr.args.clone();
                args.insert(0, *method_call_expr.receiver.clone());
                let call_arg_values =
                    self.compile_call_args_no_side_effect(builder, &args, generic_vars, ctx)?;
                let receiver_rust_ty = &call_arg_values[0].ty.rust_ty;
                let Some((_, impl_item, impl_method)) = self.modules.get_impl_item_fn(
                    receiver_rust_ty,
                    method_call_expr,
                    generic_vars,
                )?
                else {
                    return self.jit_error_result(
                        &method_call_expr.method.span(),
                        &format!("Undefined method {ident}"),
                    );
                };
                let self_ty = &*impl_item.self_ty;
                let (fn_arg_types, return_type) = get_sig_types(&impl_method.sig, Some(self_ty));

                // Closures trigger errors in compile_call_args, so only valid non-closure args are compiled.
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

                let mut call_arg_rust_tys = vec![];
                let mut arg_types: HashMap<String, TileRustType> = HashMap::new();
                for (i, param_name) in get_sig_param_names(&impl_method.sig).iter().enumerate() {
                    if i < call_arg_values.len() {
                        let call_arg_val = &call_arg_values[i];
                        let call_arg_ty = call_arg_val.ty.clone();
                        call_arg_rust_tys.push(call_arg_ty.rust_ty.clone());
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
                            method_call_expr.to_token_stream().to_string()
                        ),
                    );
                }

                let call_output_type: syn::Type =
                    generic_arg_inf.infer_type(&return_type, generic_vars);
                // If it's a cuda_tile op, it may require additional static type params.
                let mut type_params: HashMap<String, TypeParam> = HashMap::new();
                // Initialize type params with given type params.
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
                            match arg_types.get(&type_param_name) {
                                Some(arg_type) => {
                                    let cuda_tile_type_str = arg_type.get_cuda_tile_type_str();
                                    let type_instance = Some(arg_type.type_instance.clone());
                                    type_params.insert(
                                        type_param_name.to_string(),
                                        TypeParam::derive_param_from_type(
                                            type_param_name,
                                            arg_type.rust_ty.clone(),
                                            cuda_tile_type_str,
                                            type_instance,
                                        ),
                                    );
                                }
                                None => {
                                    return self.jit_error_result(
                                        &method_call_expr.method.span(),
                                        &format!("Unable to find output type: {type_param_name}"),
                                    );
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
                            method_call_expr.to_token_stream().to_string(),
                            call_output_type.to_token_stream().to_string()
                        ),
                    );
                }
                Ok(ct_type)
            }
            Expr::Call(call_expr) => {
                match &*call_expr.func {
                    Expr::Path(path_expr) => {
                        let ident = get_ident_from_path_expr(&path_expr);
                        let Some((_, fn_item)) = self.modules.functions.get(&ident.to_string())
                        else {
                            return self.jit_error_result(
                                &call_expr.func.span(),
                                &format!("Undefined function {ident}"),
                            );
                        };
                        let call_arg_values = self.compile_call_args_no_side_effect(
                            builder,
                            &call_expr.args,
                            generic_vars,
                            ctx,
                        )?;
                        let (fn_arg_types, return_type) = get_sig_types(&fn_item.sig, None);

                        // Closures trigger errors in compile_call_args, so only valid non-closure args are compiled
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
                        for (i, param_name) in get_sig_param_names(&fn_item.sig).iter().enumerate()
                        {
                            if i < call_arg_values.len() {
                                let call_arg_val = &call_arg_values[i];
                                let call_arg_ty = call_arg_val.ty.clone();
                                call_arg_rust_tys.push(call_arg_ty.rust_ty.clone());
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
                                    call_expr.to_token_stream().to_string()
                                ),
                            );
                        }

                        let call_output_type: syn::Type =
                            generic_arg_inf.infer_type(&return_type, generic_vars);
                        // If it's a cuda_tile op, it may require additional static type params.
                        let mut type_params: HashMap<String, TypeParam> = HashMap::new();
                        // Initialize type params with given type params.
                        if let Some(given_type_params) = maybe_type_params {
                            for type_param in given_type_params {
                                if let Some(name) = type_param.name() {
                                    type_params.insert(name.to_string(), type_param.clone());
                                } else {
                                    return self.jit_error_result(
                                        &call_expr.func.span(),
                                        &format!(
                                            "Failed to get name for type param {type_param:?}"
                                        ),
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
                                    match arg_types.get(&type_param_name) {
                                        Some(arg_type) => {
                                            let cuda_tile_type_str =
                                                arg_type.get_cuda_tile_type_str();
                                            type_params.insert(
                                                type_param_name.to_string(),
                                                TypeParam::derive_param_from_type(
                                                    type_param_name,
                                                    arg_type.rust_ty.clone(),
                                                    cuda_tile_type_str,
                                                    Some(arg_type.type_instance.clone()),
                                                ),
                                            );
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
                                &format!("Failed to derive output for {} \ngeneric_vars={generic_vars:#?} \ntype_params={type_params:#?}",
                                    call_expr.to_token_stream().to_string()),
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
                                call_expr.to_token_stream().to_string()
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
                        );
                    }
                }
            }
            Expr::Field(field_expr) => {
                let Some(base) =
                    self.compile_expression(builder, &field_expr.base, generic_vars, ctx, None)?
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
