/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Version and target requirements, independent of toolkit discovery or a GPU.

use crate::bytecode::{BytecodeVersion, Opcode};
use crate::ir::{Attribute, Location, Module, OpId, Operation, ScalarType, TileElementType, Type};

/// The target selected for one compilation, not the first GPU in the process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetCapabilities {
    pub bytecode_version: BytecodeVersion,
    pub architecture: String,
}

/// An unsupported feature with the location of its use, before invoking tileiras.
#[derive(Clone, Debug)]
pub struct CapabilityError {
    pub message: String,
    pub location: Box<Location>,
}

impl std::fmt::Display for CapabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CapabilityError {}

impl TargetCapabilities {
    pub fn new(bytecode_version: BytecodeVersion, architecture: impl Into<String>) -> Self {
        Self {
            bytecode_version,
            architecture: architecture.into(),
        }
    }

    /// Accept ordinary, architecture-specific, and family-specific target names.
    pub fn sm(&self) -> Option<u32> {
        self.architecture
            .strip_prefix("sm_")?
            .trim_end_matches(['a', 'f'])
            .parse()
            .ok()
    }

    fn error(&self, feature: &str, requirement: &str, location: &Location) -> CapabilityError {
        CapabilityError {
            message: format!(
                "{feature} requires {requirement}; selected Tile IR {}, target {}",
                self.bytecode_version, self.architecture
            ),
            location: Box::new(location.clone()),
        }
    }

    pub fn require_version(
        &self,
        feature: &str,
        since: BytecodeVersion,
        location: &Location,
    ) -> Result<(), CapabilityError> {
        if self.bytecode_version < since {
            Err(self.error(feature, &format!("Tile IR {since} or newer"), location))
        } else {
            Ok(())
        }
    }

    pub fn require_sm(
        &self,
        feature: &str,
        minimum: u32,
        location: &Location,
    ) -> Result<(), CapabilityError> {
        if self.sm().is_none_or(|sm| sm < minimum) {
            Err(self.error(feature, &format!("sm_{minimum} or newer"), location))
        } else {
            Ok(())
        }
    }

    fn check_scalar(
        &self,
        scalar: ScalarType,
        feature: &str,
        loc: &Location,
    ) -> Result<(), CapabilityError> {
        self.require_version(feature, scalar.minimum_version(), loc)?;
        let minimum = match scalar {
            ScalarType::F8E4M3FN | ScalarType::F8E5M2 => 90,
            ScalarType::F8E8M0FNU | ScalarType::F4E2M1FN => 100,
            ScalarType::F8E5M3FNU => 107,
            _ => return Ok(()),
        };
        self.require_sm(&format!("{feature} using {scalar:?}"), minimum, loc)
    }

    fn check_type(&self, ty: &Type, feature: &str, loc: &Location) -> Result<(), CapabilityError> {
        crate::verify_target::verify_type(ty).map_err(|message| CapabilityError {
            message: format!("{feature}: {message}"),
            location: Box::new(loc.clone()),
        })?;
        match ty {
            Type::WithPointerAttribute(base, _) => {
                self.require_version("ptr_attr", BytecodeVersion::V13_4, loc)?;
                self.check_type(base, feature, loc)
            }
            Type::Scalar(s) => self.check_scalar(*s, feature, loc),
            Type::Pointer(p) => self.check_scalar(p.pointee, feature, loc),
            Type::Tile(t) => match &t.element_type {
                TileElementType::Scalar(s) => self.check_scalar(*s, feature, loc),
                TileElementType::Pointer(p) => self.check_scalar(p.pointee, feature, loc),
            },
            Type::TensorView(t) => self.check_scalar(t.element_type, feature, loc),
            Type::PartitionView(t) => self.check_scalar(t.tensor_view.element_type, feature, loc),
            Type::GatherScatterView(t) => {
                self.require_version("gather_scatter_view", BytecodeVersion::V13_3, loc)?;
                self.check_scalar(t.tensor_view.element_type, feature, loc)
            }
            Type::StridedView(t) => {
                self.require_version("strided_view", BytecodeVersion::V13_3, loc)?;
                self.check_scalar(t.tensor_view.element_type, feature, loc)
            }
            Type::Func(f) => {
                for ty in f.inputs.iter().chain(&f.results) {
                    self.check_type(ty, feature, loc)?;
                }
                Ok(())
            }
            Type::Token => Ok(()),
        }
    }

    /// Validate reachable operations, including compiler-generated operations.
    /// Version and architecture requirements are conjunctive, never alternatives.
    pub fn validate_module(&self, module: &Module) -> Result<(), CapabilityError> {
        if module.producer.is_some() {
            self.require_version(
                "module.producer",
                BytecodeVersion::V13_3,
                &Location::Unknown,
            )?;
        }
        if self.sm() == Some(107) {
            self.require_version("target sm_107", BytecodeVersion::V13_4, &Location::Unknown)?;
        }
        if self.sm() == Some(90) {
            self.require_version("target sm_90", BytecodeVersion::V13_3, &Location::Unknown)?;
        }
        self.require_sm("Tile IR target", 80, &Location::Unknown)?;
        if !matches!(
            self.sm(),
            Some(80 | 86 | 87 | 88 | 89 | 90 | 100 | 103 | 107 | 110 | 120 | 121)
        ) {
            return Err(self.error(
                "Tile IR target",
                "an architecture supported by the public Tile IR 13.4 dialect",
                &Location::Unknown,
            ));
        }
        for global in &module.globals {
            self.check_type(&global.value.element_type, "global", &Location::Unknown)?;
            if global.constant || global.symbol_visibility != crate::ir::SymbolVisibility::Public {
                self.require_version(
                    "global constant/visibility",
                    BytecodeVersion::V13_3,
                    &Location::Unknown,
                )?;
            }
        }
        for &entry in &module.functions {
            self.check_op(module, entry, false, false)?;
        }
        Ok(())
    }

    fn check_op(
        &self,
        module: &Module,
        id: OpId,
        in_loop: bool,
        in_for: bool,
    ) -> Result<(), CapabilityError> {
        let op = module.op(id);
        let feature = format!("cuda_tile.{}", op.opcode.name());
        let loc = &op.location;
        self.require_version(&feature, op.opcode.minimum_version(), loc)?;
        for ty in op
            .result_types
            .iter()
            .chain(op.operands.iter().map(|v| module.value_type(*v)))
        {
            self.check_type(ty, &feature, loc)?;
        }
        for (name, attr) in &op.attributes {
            match attr {
                Attribute::Type(ty) => self.check_type(ty, &feature, loc)?,
                Attribute::DenseElements(d) => self.check_type(&d.element_type, &feature, loc)?,
                Attribute::OptimizationHints(hints) => {
                    for (arch, values) in &hints.entries {
                        if !matches!(
                            arch.as_str(),
                            "default"
                                | "sm_80"
                                | "sm_86"
                                | "sm_87"
                                | "sm_88"
                                | "sm_89"
                                | "sm_90"
                                | "sm_100"
                                | "sm_103"
                                | "sm_107"
                                | "sm_110"
                                | "sm_120"
                                | "sm_121"
                        ) {
                            return Err(self.error(
                                &format!("optimization hint target {arch}"),
                                "a public Tile IR architecture key or default",
                                loc,
                            ));
                        }
                        if arch == "sm_90" {
                            self.require_version(
                                "optimization hint target sm_90",
                                BytecodeVersion::V13_3,
                                loc,
                            )?;
                        }
                        if arch == "sm_107" {
                            self.require_version(
                                "optimization hint target sm_107",
                                BytecodeVersion::V13_4,
                                loc,
                            )?;
                        }
                        for (key, value) in values {
                            if key == "num_worker_warps_per_cta" {
                                self.require_version(key, BytecodeVersion::V13_3, loc)?;
                            }
                            let mut hint_sm =
                                arch.strip_prefix("sm_").and_then(|s| s.parse::<u32>().ok());
                            if arch == "default"
                                && !hints.entries.iter().any(|(target, values)| {
                                    target == &self.architecture
                                        && values.iter().any(|(key, _)| key == "num_cta_in_cga")
                                })
                            {
                                hint_sm = self.sm();
                            }
                            if key == "num_cta_in_cga"
                                && hint_sm.is_some_and(|sm| sm < 90)
                                && matches!(value, Attribute::Integer(n, _) if *n != 1)
                            {
                                return Err(self.error(
                                    &feature,
                                    "num_cta_in_cga=1 on targets below sm_90",
                                    loc,
                                ));
                            }
                            let valid = match (key.as_str(), value) {
                                ("num_cta_in_cga", Attribute::Integer(n, _)) => {
                                    op.opcode == Opcode::Entry && matches!(n, 1 | 2 | 4 | 8 | 16)
                                }
                                ("occupancy", Attribute::Integer(n, _)) => {
                                    op.opcode == Opcode::Entry && (1..=32).contains(n)
                                }
                                ("num_worker_warps_per_cta", Attribute::Integer(n, _)) => {
                                    op.opcode == Opcode::Entry
                                        && if self.bytecode_version >= BytecodeVersion::V13_4 {
                                            matches!(n, 4 | 8)
                                        } else {
                                            matches!(n, 1 | 2 | 4 | 8 | 16 | 32)
                                        }
                                }
                                ("allow_tma", Attribute::Bool(_)) => {
                                    matches!(op.opcode, Opcode::LoadViewTko | Opcode::StoreViewTko)
                                }
                                ("latency", Attribute::Integer(n, _)) => {
                                    matches!(
                                        op.opcode,
                                        Opcode::LoadViewTko
                                            | Opcode::StoreViewTko
                                            | Opcode::LoadPtrTko
                                            | Opcode::StorePtrTko
                                    ) && (1..=10).contains(n)
                                }
                                _ => false,
                            };
                            if !valid {
                                return Err(CapabilityError {
                                    message: format!(
                                        "{feature}: invalid optimization hint {key}={value:?}"
                                    ),
                                    location: Box::new(loc.clone()),
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
            let since = match (name.as_str(), attr) {
                ("rounding_mode", Attribute::Integer(7, _)) => Some(BytecodeVersion::V13_4),
                ("saturating", Attribute::Bool(true)) => Some(BytecodeVersion::V13_4),
                ("inbounds", Attribute::Array(a))
                    if a.iter().any(|v| matches!(v, Attribute::Bool(true))) =>
                {
                    Some(BytecodeVersion::V13_4)
                }
                ("fast_acc", Attribute::Bool(true)) => Some(BytecodeVersion::V13_3),
                ("producer", _) => Some(BytecodeVersion::V13_3),
                ("rounding_mode", Attribute::Integer(n, _))
                    if op.opcode == Opcode::Exp && *n != 5 =>
                {
                    Some(BytecodeVersion::V13_3)
                }
                _ => None,
            };
            if let Some(since) = since {
                self.require_version(&format!("{feature}.{name}"), since, loc)?;
            }
        }
        if op.opcode == Opcode::MmaFScaled {
            self.require_sm(&feature, 100, loc)?;
            let new_scale = op.operands.iter().skip(3).any(|v| {
                matches!(module.value_type(*v), Type::Tile(t) if t.element_type == TileElementType::Scalar(ScalarType::F8E5M3FNU))
            });
            // This is a configuration/family restriction, not sm >= 107:
            // SM120/121 support NVFP4 but not this new scale format.
            if new_scale && self.sm() != Some(107) {
                return Err(self.error(
                    "cuda_tile.mmaf_scaled with f8E5M3FNU scales",
                    "sm_107",
                    loc,
                ));
            }
        }
        if matches!(op.opcode, Opcode::AtomicRMW | Opcode::AtomicRedViewTko)
            && integer_attr(op, "mode") == Some(4)
            && op.operands.iter().any(|v| matches!(module.value_type(*v), Type::Tile(t) if t.element_type == TileElementType::Scalar(ScalarType::BF16))) {
            self.require_version(&format!("{feature} with bf16 add"), BytecodeVersion::V13_3, loc)?;
            self.require_sm(&format!("{feature} with bf16 add"), 90, loc)?;
        }
        if op.opcode == Opcode::Return && in_loop {
            self.require_version("return inside loop", BytecodeVersion::V13_4, loc)?;
        }
        if op.opcode == Opcode::Return && in_for {
            return Err(self.error("return inside for", "return outside the for region", loc));
        }
        crate::verify_target::verify_operation(module, op, self)?;
        for &region in &op.regions {
            for &block in &module.region(region).blocks {
                for &child in &module.block(block).ops {
                    self.check_op(
                        module,
                        child,
                        in_loop || op.opcode == Opcode::Loop,
                        in_for || op.opcode == Opcode::For,
                    )?;
                }
            }
        }
        Ok(())
    }
}

fn integer_attr(op: &Operation, name: &str) -> Option<i64> {
    op.attributes.iter().find_map(|(key, value)| match value {
        Attribute::Integer(n, _) if key == name => Some(*n),
        _ => None,
    })
}
