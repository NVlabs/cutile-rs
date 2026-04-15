/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Optimization hint types shared between both compiler backends.
//!
//! Pure Rust — no melior or tile-ir dependency.

use crate::ast::SourceLocation;
use crate::error::{JITError, SpannedJITError};
use quote::ToTokens;
use std::collections::BTreeMap;
use syn::{Expr, Lit};

/// Per-architecture (SM) optimization hints for kernel compilation.
///
/// `allow_tma` and `latency` are per-op hints (passed at load/store call sites),
/// not entry-level hints. Setting them at the entry level is an error.
pub struct SMHints {
    pub gpu_name: String,
    pub num_cta_in_cga: Option<i32>,
    pub occupancy: Option<i32>,
    pub max_divisibility: Option<i32>,
}

impl SMHints {
    pub fn new(gpu_name: String) -> Self {
        Self {
            gpu_name,
            num_cta_in_cga: None,
            occupancy: None,
            max_divisibility: None,
        }
    }

    pub fn set_num_cta_in_cga(&mut self, hint: &Expr) -> Result<(), JITError> {
        if self.num_cta_in_cga.is_some() {
            return SourceLocation::unknown()
                .jit_error_result("num_cta_in_cga hint has already been set");
        }
        self.num_cta_in_cga = Some(get_int_hint(hint)?);
        Ok(())
    }

    pub fn set_occupancy(&mut self, hint: &Expr) -> Result<(), JITError> {
        if self.occupancy.is_some() {
            return SourceLocation::unknown()
                .jit_error_result("occupancy hint has already been set");
        }
        self.occupancy = Some(get_int_hint(hint)?);
        Ok(())
    }

    pub fn set_max_divisibility(&mut self, hint: &Expr) -> Result<(), JITError> {
        if self.max_divisibility.is_some() {
            return SourceLocation::unknown()
                .jit_error_result("max_divisibility hint has already been set");
        }
        self.max_divisibility = Some(get_int_hint(hint)?);
        Ok(())
    }
}

fn get_int_hint(expr: &Expr) -> Result<i32, JITError> {
    let Expr::Lit(lit) = expr else {
        return SourceLocation::unknown()
            .jit_error_result("expected a literal value for optimization hint");
    };
    let Lit::Int(int_expr) = &lit.lit else {
        return SourceLocation::unknown()
            .jit_error_result("expected an integer literal for optimization hint");
    };
    int_expr
        .base10_parse()
        .map_err(|e| JITError::Generic(format!("Failed to parse int hint: {e}")))
}

/// Runtime compile options for kernel JIT compilation.
///
/// These options control kernel-level compilation hints that can vary between
/// launches. Different values trigger separate JIT compilations (they are part
/// of the cache key).
#[derive(Debug, Eq, PartialEq, Hash, Clone, Default)]
pub struct CompileOptions {
    pub occupancy: Option<i32>,
    pub num_cta_in_cga: Option<i32>,
    pub max_divisibility: Option<i32>,
}

impl CompileOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn occupancy(mut self, occupancy: i32) -> Self {
        self.occupancy = Some(occupancy);
        self
    }

    pub fn num_cta_in_cga(mut self, num_cta_in_cga: i32) -> Self {
        self.num_cta_in_cga = Some(num_cta_in_cga);
        self
    }

    pub fn max_divisibility(mut self, max_divisibility: i32) -> Self {
        self.max_divisibility = Some(max_divisibility);
        self
    }
}

/// Collection of optimization hints for kernel compilation, keyed by SM architecture.
pub struct OptimizationHints {
    pub target_gpu_name: Option<String>,
    pub tile_as_hints: BTreeMap<String, SMHints>,
}

impl OptimizationHints {
    pub fn empty() -> OptimizationHints {
        Self {
            target_gpu_name: None,
            tile_as_hints: BTreeMap::new(),
        }
    }

    fn parse_key_value(expr: &Expr) -> Result<(String, Expr), JITError> {
        let Expr::Assign(key_val) = expr else {
            return SourceLocation::unknown()
                .jit_error_result("expected an assignment expression in optimization hints");
        };
        let Expr::Path(key_path) = &*key_val.left else {
            return SourceLocation::unknown().jit_error_result(
                "Expected path expression on LHS of optimization hints assignment.",
            );
        };
        if key_path.path.segments.len() != 1 {
            return SourceLocation::unknown().jit_error_result(&format!(
                "Expected single-segment path in optimization hints key, got {} segments.",
                key_path.path.segments.len()
            ));
        }
        let key = key_path.path.segments.last().unwrap().ident.to_string();
        let value = *key_val.right.clone();
        Ok((key, value))
    }

    pub fn parse(expr: &Expr, target_gpu_name: String) -> Result<OptimizationHints, JITError> {
        let Expr::Tuple(opt_hints) = expr else {
            return SourceLocation::unknown()
                .jit_error_result("expected a tuple expression for optimization hints");
        };
        let mut result = OptimizationHints::empty();
        result.target_gpu_name = Some(target_gpu_name);
        for sm_key_val in &opt_hints.elems {
            let (opt_key, opt_value) = Self::parse_key_value(sm_key_val)?;
            {
                if !opt_key.starts_with("sm_") {
                    return SourceLocation::unknown().jit_error_result(&format!(
                        "Unexpected optimization hint {}.",
                        sm_key_val.to_token_stream()
                    ));
                }
                let Expr::Tuple(hints_tuple) = opt_value else {
                    return SourceLocation::unknown().jit_error_result(
                        "expected a tuple expression for architecture-specific optimization hints",
                    );
                };
                let mut sm_hints_result = SMHints::new(opt_key.clone());
                for hint_key_val in hints_tuple.elems.iter() {
                    let (key, hints) = Self::parse_key_value(hint_key_val)?;
                    match key.as_str() {
                        "num_cta_in_cga" => sm_hints_result.set_num_cta_in_cga(&hints)?,
                        "occupancy" => sm_hints_result.set_occupancy(&hints)?,
                        "max_divisibility" => sm_hints_result.set_max_divisibility(&hints)?,
                        "allow_tma" | "latency" => {
                            return SourceLocation::unknown().jit_error_result(&format!(
                                "'{key}' is a per-op hint and cannot be set at the entry level. \
                                 Use it as a parameter on individual load/store operations instead."
                            ));
                        }
                        _ => {
                            return SourceLocation::unknown().jit_error_result(&format!(
                                "Unexpected optimization hint key '{key}'."
                            ));
                        }
                    }
                }
                if result
                    .tile_as_hints
                    .insert(opt_key.clone(), sm_hints_result)
                    .is_some()
                {
                    return SourceLocation::unknown().jit_error_result(&format!(
                        "Duplicate optimization hint key '{opt_key}'."
                    ));
                }
            }
        }
        Ok(result)
    }

    pub fn get_sm_hints(&self, key: &str) -> Option<&SMHints> {
        self.tile_as_hints.get(key)
    }

    /// Applies runtime compile options, overriding entry-level hints.
    pub fn apply_compile_options(&mut self, options: &CompileOptions) {
        if options.occupancy.is_none()
            && options.num_cta_in_cga.is_none()
            && options.max_divisibility.is_none()
        {
            return;
        }
        let target_arch = self
            .target_gpu_name
            .clone()
            .unwrap_or_else(|| "sm_100".to_string());
        let sm_hints = self
            .tile_as_hints
            .entry(target_arch.clone())
            .or_insert_with(|| SMHints::new(target_arch));
        if let Some(occupancy) = options.occupancy {
            sm_hints.occupancy = Some(occupancy);
        }
        if let Some(num_cta_in_cga) = options.num_cta_in_cga {
            sm_hints.num_cta_in_cga = Some(num_cta_in_cga);
        }
        if let Some(max_divisibility) = options.max_divisibility {
            sm_hints.max_divisibility = Some(max_divisibility);
        }
    }
}
