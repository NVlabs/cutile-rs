/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[derive(Debug, Clone)]
pub enum ValidParamType {
    Scalar(ScalarParamType),
    Pointer(PointerParamType),
    Tensor(TensorParamType),
}

#[derive(Debug, Clone)]
pub struct ScalarParamType {
    pub element_type: String,
}

#[derive(Debug, Clone)]
pub struct PointerParamType {
    pub mutable: bool,
    pub element_type: String,
}

#[derive(Debug, Clone)]
pub struct TensorParamType {
    pub element_type: String,
    pub shape: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct Validator {
    pub params: Vec<ValidParamType>,
}
