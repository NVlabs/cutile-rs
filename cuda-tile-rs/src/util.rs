/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use melior::{
    Context, StringRef,
    ir::{Attribute, Identifier, Operation, Type, operation::OperationLike},
    pass::PassManager,
};
use mlir_sys::mlirPassManagerRunOnOp;
use mlir_sys::{mlirAttributeParseGet, mlirOperationCreateParse};
use std::ffi::CString;

use crate::cuda_tile::ModuleOperation;

pub fn operation_parse<'c>(
    context: &'c Context,
    source: &str,
    source_name: Option<&str>,
) -> Option<Operation<'c>> {
    let source = CString::new(source).unwrap();
    let source = StringRef::from_c_str(&source);
    let source_name = CString::new(source_name.unwrap_or("sourceName")).unwrap();
    let source_name_ref = StringRef::from_c_str(&source_name);
    unsafe {
        Operation::from_option_raw(mlirOperationCreateParse(
            context.to_raw(),
            source.to_raw(),
            source_name_ref.to_raw(),
        ))
    }
}

pub fn type_parse<'c>(context: &'c Context, source: &str) -> Option<Type<'c>> {
    Type::parse(context, source)
}

pub fn attribute_parse<'c>(context: &'c Context, source: &str) -> Option<Attribute<'c>> {
    unsafe {
        Attribute::from_option_raw(mlirAttributeParseGet(
            context.to_raw(),
            StringRef::new(source).to_raw(),
        ))
    }
}

pub fn parse_named_attr<'c>(
    context: &'c Context,
    name: &str,
    attr_str: &str,
) -> (Identifier<'c>, Attribute<'c>) {
    let Some(attr) = Attribute::parse(context, attr_str) else {
        panic!("Failed to parse named attribute {name} = {attr_str}");
    };
    (Identifier::new(context, name), attr)
}

pub fn execute_pass_manager(
    pass_manager: &mut PassManager,
    module_op: &mut ModuleOperation,
) -> Result<(), melior::Error> {
    let result =
        unsafe { mlirPassManagerRunOnOp(pass_manager.to_raw(), module_op.as_operation().to_raw()) };
    if result.value == 1 {
        Ok(())
    } else {
        Err(melior::Error::RunPass)
    }
}
