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
use syn::spanned::Spanned;
use syn::{Expr, Lit};

// ---------------------------------------------------------------------------
// Hint value domains
// ---------------------------------------------------------------------------

/// The value domain of one optimization hint, as constrained by the Tile IR
/// attribute definitions.
///
/// This is the single place that knows a hint's legal values. The macro path
/// ([`SMHints`] setters), the runtime builder ([`CompileOptions`]) and the
/// compile boundary all validate through it, so a value can never be accepted
/// on one path and rejected on another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HintDomain {
    /// Any integer in the inclusive range, e.g. `occupancy` in `[1, 32]`.
    InclusiveRange { min: i32, max: i32 },
    /// A power of two in the inclusive range, e.g. `num_cta_in_cga` in
    /// `[1, 16]`.
    PowerOfTwoInclusiveRange { min: i32, max: i32 },
}

impl HintDomain {
    /// Human-readable rendering of the accepted values, used verbatim in
    /// diagnostics so an error always states the range it enforced.
    pub fn describe(self) -> String {
        match self {
            HintDomain::InclusiveRange { min, max } => {
                format!("an integer in [{min}, {max}]")
            }
            HintDomain::PowerOfTwoInclusiveRange { min, max } => {
                format!("a power of two in [{min}, {max}]")
            }
        }
    }

    /// `true` when `value` is inside the domain.
    pub fn accepts(self, value: i32) -> bool {
        // `i32` has no `is_power_of_two`; only the unsigned integers do. A
        // non-positive value can never be a member of a power-of-two domain,
        // so filter on the sign first and then use `unsigned_abs`.
        let is_power_of_two = |v: i32| v > 0 && v.unsigned_abs().is_power_of_two();
        match self {
            HintDomain::InclusiveRange { .. } => value >= self.min() && value <= self.max(),
            HintDomain::PowerOfTwoInclusiveRange { .. } => {
                is_power_of_two(value) && value >= self.min() && value <= self.max()
            }
        }
    }

    fn min(self) -> i32 {
        match self {
            HintDomain::InclusiveRange { min, .. }
            | HintDomain::PowerOfTwoInclusiveRange { min, .. } => min,
        }
    }

    fn max(self) -> i32 {
        match self {
            HintDomain::InclusiveRange { max, .. }
            | HintDomain::PowerOfTwoInclusiveRange { max, .. } => max,
        }
    }
}

/// Tile IR constraint for the entry-level `occupancy` hint.
pub const OCCUPANCY_DOMAIN: HintDomain = HintDomain::InclusiveRange { min: 1, max: 32 };

/// Tile IR constraint for the entry-level `num_cta_in_cga` hint.
pub const NUM_CTA_IN_CGA_DOMAIN: HintDomain =
    HintDomain::PowerOfTwoInclusiveRange { min: 1, max: 16 };

/// Tile IR constraint for the entry-level `num_worker_warps_per_cta` hint.
///
/// The value domain is independent of the bytecode-version gate: the writer
/// rejects this hint outright below bytecode 13.3 (see
/// `cutile-ir::bytecode::writer`), so a valid value can still be unusable on an
/// older toolkit. That is a version constraint, not a value constraint.
pub const NUM_WORKER_WARPS_PER_CTA_DOMAIN: HintDomain =
    HintDomain::PowerOfTwoInclusiveRange { min: 1, max: 32 };

/// Tile IR constraint for the per-op `latency` hint.
pub const LATENCY_DOMAIN: HintDomain = HintDomain::InclusiveRange { min: 1, max: 10 };

/// A hint value that violates its [`HintDomain`].
///
/// The message states the rejected value, the hint name, and the accepted
/// range, because that is what the user needs in order to fix the call. It
/// deliberately carries no source location: the value-domain check is pure
/// Rust and has no business inventing one. Callers attach the position they
/// actually hold — the macro path resolves the offending expression's span
/// through [`HintLocationResolver`], while the runtime builder has no span at
/// all and reports the `CompileOptions` field name instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HintValueError {
    /// Hint name exactly as it is written in source, or as the
    /// `CompileOptions` field name on the runtime path.
    pub hint: &'static str,
    /// The rejected value.
    pub value: i32,
    /// The domain the value was checked against.
    pub domain: HintDomain,
}

impl HintValueError {
    /// The complete diagnostic, with no location prefix.
    pub fn message(&self) -> String {
        format!(
            "invalid value {} for optimization hint '{}': expected {}",
            self.value,
            self.hint,
            self.domain.describe()
        )
    }

    /// Turn this into a `JITError`, locating it when `location` is known.
    pub fn into_jit_error(self, location: SourceLocation) -> JITError {
        match location.is_known() {
            true => JITError::Located(self.message(), location),
            false => JITError::Generic(self.message()),
        }
    }
}

impl std::fmt::Display for HintValueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for HintValueError {}

/// Resolves a `syn` span inside a module's AST to an absolute source
/// location.
///
/// `crate::hints` is pure Rust by design (see the module docs), so it cannot
/// reach the compiler's `SpanBase`-backed resolver. The resolver is a
/// parameter instead, and the identity implementation below is what callers
/// get when no real source text is available.
pub trait HintLocationResolver {
    /// Absolute location of `span`, or an unknown location when the span
    /// carries no real source information.
    fn resolve(&self, span: proc_macro2::Span) -> SourceLocation;
}

/// Resolver used when the caller has no source-text anchor.
///
/// `proc_macro2` runs in fallback mode at JIT time and reports every span as
/// `call_site`, so this returns [`SourceLocation::unknown`] rather than a
/// fabricated `1:0` position.
pub struct UnknownLocation;

impl HintLocationResolver for UnknownLocation {
    fn resolve(&self, _span: proc_macro2::Span) -> SourceLocation {
        SourceLocation::unknown()
    }
}

/// Checks `value` against `domain` for the hint named `hint`.
///
/// This is the single value-domain gate. Every entry point validates through
/// it, so a value can never be accepted on one path and rejected on another.
pub fn check_hint_value(
    hint: &'static str,
    value: i32,
    domain: HintDomain,
) -> Result<i32, HintValueError> {
    match domain.accepts(value) {
        true => Ok(value),
        false => Err(HintValueError {
            hint,
            value,
            domain,
        }),
    }
}

/// Per-architecture (SM) optimization hints for kernel compilation.
///
/// `allow_tma` and `latency` are per-op hints (passed at load/store call sites),
/// not entry-level hints. Setting them at the entry level is an error.
pub struct SMHints {
    pub gpu_name: String,
    pub num_cta_in_cga: Option<i32>,
    pub occupancy: Option<i32>,
    pub max_divisibility: Option<i32>,
    pub num_worker_warps_per_cta: Option<i32>,
}

impl SMHints {
    pub fn new(gpu_name: String) -> Self {
        Self {
            gpu_name,
            num_cta_in_cga: None,
            occupancy: None,
            max_divisibility: None,
            num_worker_warps_per_cta: None,
        }
    }

    /// Shared body of the entry-level setters: reject a duplicate key, parse
    /// the integer literal, then check it against its Tile IR value domain.
    ///
    /// The value check happens here — before the value is stored and long
    /// before a backend sees it — so an out-of-range hint is a cutile error
    /// tied to the offending expression rather than a late `tileiras`
    /// verifier failure.
    fn set_entry_hint(
        slot: &mut Option<i32>,
        hint: &'static str,
        domain: HintDomain,
        value_expr: &Expr,
        resolver: &dyn HintLocationResolver,
    ) -> Result<(), JITError> {
        if slot.is_some() {
            return Err(located_or_generic(
                &format!("{hint} hint has already been set"),
                resolver.resolve(value_expr.span()),
            ));
        }
        let value = get_int_hint(value_expr, resolver)?;
        *slot = Some(
            check_hint_value(hint, value, domain)
                .map_err(|err| err.into_jit_error(resolver.resolve(value_expr.span())))?,
        );
        Ok(())
    }

    pub fn set_num_cta_in_cga(
        &mut self,
        hint: &Expr,
        resolver: &dyn HintLocationResolver,
    ) -> Result<(), JITError> {
        Self::set_entry_hint(
            &mut self.num_cta_in_cga,
            "num_cta_in_cga",
            NUM_CTA_IN_CGA_DOMAIN,
            hint,
            resolver,
        )
    }

    pub fn set_occupancy(
        &mut self,
        hint: &Expr,
        resolver: &dyn HintLocationResolver,
    ) -> Result<(), JITError> {
        Self::set_entry_hint(
            &mut self.occupancy,
            "occupancy",
            OCCUPANCY_DOMAIN,
            hint,
            resolver,
        )
    }

    pub fn set_max_divisibility(
        &mut self,
        hint: &Expr,
        resolver: &dyn HintLocationResolver,
    ) -> Result<(), JITError> {
        // `max_divisibility` is a ceiling on auto-inferred divisibility rather
        // than a Tile IR attribute with a fixed legal set, so it is parsed but
        // not range-checked here.
        if self.max_divisibility.is_some() {
            return Err(located_or_generic(
                "max_divisibility hint has already been set",
                resolver.resolve(hint.span()),
            ));
        }
        self.max_divisibility = Some(get_int_hint(hint, resolver)?);
        Ok(())
    }

    pub fn set_num_worker_warps_per_cta(
        &mut self,
        hint: &Expr,
        resolver: &dyn HintLocationResolver,
    ) -> Result<(), JITError> {
        Self::set_entry_hint(
            &mut self.num_worker_warps_per_cta,
            "num_worker_warps_per_cta",
            NUM_WORKER_WARPS_PER_CTA_DOMAIN,
            hint,
            resolver,
        )
    }
}

fn get_int_hint(expr: &Expr, resolver: &dyn HintLocationResolver) -> Result<i32, JITError> {
    let location = || resolver.resolve(expr.span());
    let Expr::Lit(lit) = expr else {
        return Err(located_or_generic(
            "expected a literal value for optimization hint",
            location(),
        ));
    };
    let Lit::Int(int_expr) = &lit.lit else {
        return Err(located_or_generic(
            "expected an integer literal for optimization hint",
            location(),
        ));
    };
    int_expr
        .base10_parse()
        .map_err(|e| JITError::Generic(format!("Failed to parse int hint: {e}")))
}

/// Builds a located error when `location` is real, and a plain one otherwise.
///
/// Keeps the pre-existing diagnostic shape: a hint parsed from a synthetic AST
/// (no `SpanBase`, so no file) still reports `Generic` rather than a `Located`
/// variant pointing at an empty file and line 0.
fn located_or_generic(message: &str, location: SourceLocation) -> JITError {
    match location.is_known() {
        true => JITError::Located(message.to_string(), location),
        false => JITError::Generic(message.to_string()),
    }
}

/// Device debug information to request from `tileiras`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugInfoLevel {
    /// No device debug information.
    None,
    /// Source line tables for profiling optimized kernels.
    Line,
    /// Source locations and inline frames for debugging unoptimized kernels.
    /// Source-variable inspection depends on support in the Tile IR toolchain.
    Full,
}
/// Runtime compile options for kernel JIT compilation.
///
/// These options control kernel-level compilation hints that can vary between
/// launches. Different values trigger separate JIT compilations (they are part
/// of the cache key).
///
/// `new()` and `default()` follow Cargo's profile `debug` setting for the
/// target `cutile-compiler` library: disabled selects `None`, enabled selects
/// `Full`. Cargo exposes only on/off to build scripts, so this also selects
/// `Full` for `debug = "line-tables-only"` or `"limited"`, not `Line`.
/// `CUDA_RUST_DEBUG=none|line|full` overrides this default at build time;
/// changing it when running an already-built app has no effect. Use
/// [`Self::debug_info`] to override the default for one compilation.
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct CompileOptions {
    pub occupancy: Option<i32>,
    pub num_cta_in_cga: Option<i32>,
    pub max_divisibility: Option<i32>,
    pub num_worker_warps_per_cta: Option<i32>,
    /// `tileiras` optimization level (`--opt-level`). `None` means the
    /// default: 3, or 0 when `device_debug` is set.
    pub opt_level: Option<u8>,
    /// Compile for debugging (`tileiras --device-debug`): the frontend stops
    /// hoisting bounds checks out of loops, so every check that runs on the
    /// device sits at the source line that wrote it, and the backend
    /// generates debug information. Checks the compiler discharged by proof
    /// or moved to launch time never reach device code in any mode. Implies
    /// `--opt-level 0` unless `opt_level` is set explicitly.
    pub device_debug: bool,
    /// Emit line-number information (`tileiras --lineinfo`) for profiler
    /// correlation, without the rest of the debug contract.
    pub lineinfo: bool,
    /// Instrument memory accesses for Compute Sanitizer's memcheck tool
    /// (`tileiras --sanitize=memcheck`).
    pub sanitize_memcheck: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            occupancy: None,
            num_cta_in_cga: None,
            max_divisibility: None,
            num_worker_warps_per_cta: None,
            opt_level: None,
            device_debug: env!("CUTILE_BUILD_DEBUG_INFO") == "full",
            lineinfo: env!("CUTILE_BUILD_DEBUG_INFO") == "line",
            sanitize_memcheck: false,
        }
    }
}

impl CompileOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects exactly one debug-info mode, replacing both debug flags.
    ///
    /// Other options, including an explicitly set optimization level, are
    /// preserved. With no explicit optimization level, `Full` uses level 0
    /// and `None` / `Line` use level 3.
    pub fn debug_info(mut self, level: DebugInfoLevel) -> Self {
        self.device_debug = level == DebugInfoLevel::Full;
        self.lineinfo = level == DebugInfoLevel::Line;
        self
    }

    /// Sets the `occupancy` hint. Rejects values outside `[1, 32]`.
    ///
    /// Returns `Result` rather than `Self` so that a bad value surfaces at the
    /// call which wrote it instead of at first launch. Chaining still reads
    /// the same way where the caller propagates with `?`.
    pub fn occupancy(self, occupancy: i32) -> Result<Self, HintValueError> {
        Ok(Self {
            occupancy: Some(check_hint_value("occupancy", occupancy, OCCUPANCY_DOMAIN)?),
            ..self
        })
    }

    /// Sets the `num_cta_in_cga` hint. Rejects values that are not a power of
    /// two in `[1, 16]`.
    pub fn num_cta_in_cga(self, num_cta_in_cga: i32) -> Result<Self, HintValueError> {
        Ok(Self {
            num_cta_in_cga: Some(check_hint_value(
                "num_cta_in_cga",
                num_cta_in_cga,
                NUM_CTA_IN_CGA_DOMAIN,
            )?),
            ..self
        })
    }

    /// Sets `max_divisibility`.
    ///
    /// Not range-checked: it is a ceiling on the compiler's auto-inferred
    /// divisibility facts, not a Tile IR attribute with a fixed legal set.
    pub fn max_divisibility(self, max_divisibility: i32) -> Result<Self, HintValueError> {
        Ok(Self {
            max_divisibility: Some(max_divisibility),
            ..self
        })
    }

    /// Sets the `num_worker_warps_per_cta` hint. Rejects values that are not a
    /// power of two in `[1, 32]`.
    ///
    /// A valid value here can still be unusable on the active toolchain: the
    /// hint requires Tile IR bytecode 13.3 or newer, which the bytecode writer
    /// enforces separately. This method checks the value only.
    pub fn num_worker_warps_per_cta(
        self,
        num_worker_warps_per_cta: i32,
    ) -> Result<Self, HintValueError> {
        Ok(Self {
            num_worker_warps_per_cta: Some(check_hint_value(
                "num_worker_warps_per_cta",
                num_worker_warps_per_cta,
                NUM_WORKER_WARPS_PER_CTA_DOMAIN,
            )?),
            ..self
        })
    }

    pub fn opt_level(mut self, opt_level: u8) -> Self {
        self.opt_level = Some(opt_level);
        self
    }

    pub fn device_debug(mut self, device_debug: bool) -> Self {
        self.device_debug = device_debug;
        self
    }

    pub fn lineinfo(mut self, lineinfo: bool) -> Self {
        self.lineinfo = lineinfo;
        self
    }

    pub fn sanitize_memcheck(mut self, sanitize_memcheck: bool) -> Self {
        self.sanitize_memcheck = sanitize_memcheck;
        self
    }

    /// Validates every hint value held by this `CompileOptions`.
    ///
    /// The hint fields are public and the builder is not the only way to fill
    /// them, so a struct literal or a direct field assignment can bypass the
    /// per-setter checks. The compile boundary calls this, which is what makes
    /// "an out-of-range value never reaches the backend" true for every
    /// construction path rather than only for the builder.
    pub fn validate(&self) -> Result<(), HintValueError> {
        if let Some(value) = self.occupancy {
            check_hint_value("occupancy", value, OCCUPANCY_DOMAIN)?;
        }
        if let Some(value) = self.num_cta_in_cga {
            check_hint_value("num_cta_in_cga", value, NUM_CTA_IN_CGA_DOMAIN)?;
        }
        if let Some(value) = self.num_worker_warps_per_cta {
            check_hint_value(
                "num_worker_warps_per_cta",
                value,
                NUM_WORKER_WARPS_PER_CTA_DOMAIN,
            )?;
        }
        Ok(())
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

    pub fn parse(
        expr: &Expr,
        target_gpu_name: String,
        resolver: &dyn HintLocationResolver,
    ) -> Result<OptimizationHints, JITError> {
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
                        "num_cta_in_cga" => sm_hints_result.set_num_cta_in_cga(&hints, resolver)?,
                        "occupancy" => sm_hints_result.set_occupancy(&hints, resolver)?,
                        "max_divisibility" => {
                            sm_hints_result.set_max_divisibility(&hints, resolver)?
                        }
                        "num_worker_warps_per_cta" => {
                            sm_hints_result.set_num_worker_warps_per_cta(&hints, resolver)?
                        }
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
    ///
    /// The options are re-validated here. Their hint fields are public, so a
    /// struct literal or a direct field assignment can bypass the checked
    /// builder; this is the point that catches such a value, and it runs
    /// before any hint reaches the IR or the JIT cache key.
    pub fn apply_compile_options(&mut self, options: &CompileOptions) -> Result<(), JITError> {
        options
            .validate()
            .map_err(|err| err.into_jit_error(SourceLocation::unknown()))?;
        if options.occupancy.is_none()
            && options.num_cta_in_cga.is_none()
            && options.max_divisibility.is_none()
            && options.num_worker_warps_per_cta.is_none()
        {
            return Ok(());
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
        if let Some(num_worker_warps_per_cta) = options.num_worker_warps_per_cta {
            sm_hints.num_worker_warps_per_cta = Some(num_worker_warps_per_cta);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolver that reports a fixed, known location, so tests can assert that
    /// a diagnostic is *located* rather than merely present.
    struct FixedLocation;

    impl HintLocationResolver for FixedLocation {
        fn resolve(&self, _span: proc_macro2::Span) -> SourceLocation {
            SourceLocation::new("kernel.rs".to_string(), 7, 4)
        }
    }

    /// Resolver that reports no location, standing in for a synthetic AST.
    struct NoLocation;

    impl HintLocationResolver for NoLocation {
        fn resolve(&self, _span: proc_macro2::Span) -> SourceLocation {
            SourceLocation::unknown()
        }
    }

    fn parse_hints(
        expr: &Expr,
        resolver: &dyn HintLocationResolver,
    ) -> Result<OptimizationHints, JITError> {
        OptimizationHints::parse(expr, "sm_89".to_string(), resolver)
    }

    /// Builds the expression `OptimizationHints::parse` actually receives.
    ///
    /// The attribute is written `optimization_hints = (<arch> = (<hints>,),)`,
    /// but `parse` is handed the attribute's *value*, i.e. the right-hand side
    /// of the assignment: the tuple `(<hints>,)`. The trailing comma matters —
    /// without it `(occupancy = 33)` is a parenthesised assignment rather than
    /// an `Expr::Tuple`, and the parser rejects it before reaching a value
    /// check. Mirroring that shape here keeps the fixtures honest.
    fn hints_expr(body: &str) -> Expr {
        // Builds exactly what `OptimizationHints::parse` is handed for `sm_89`:
        // the attribute value `(sm_89 = (<hints>,),)`. `body` is the inner hint
        // list, e.g. `"occupancy = 33"` or `"occupancy = 4, num_cta_in_cga = 2"`.
        // The single-element trailing comma is load-bearing: `syn` only yields
        // `Expr::Tuple` for `(x,)`, and without it `parse` rejects the fixture
        // before any value check runs.
        let body = match body.trim_end().ends_with(',') {
            true => body.trim_end().to_string(),
            false => format!("{},", body.trim_end()),
        };
        syn::parse_str(&format!("(sm_89 = ({body}),)")).expect("hint tuple parses")
    }

    /// `Result::expect_err` needs `T: Debug`, and `OptimizationHints` carries
    /// no `Debug` impl (deliberately — it holds a full hint map). This flips
    /// the result and panics on the success arm with a static message.
    fn hints_error(result: Result<OptimizationHints, JITError>, what: &str) -> JITError {
        match result {
            Ok(_) => panic!("{what}"),
            Err(err) => err,
        }
    }

    fn error_text(err: &JITError) -> String {
        format!("{err}")
    }

    // -- Value domains ----------------------------------------------------

    #[test]
    fn occupancy_domain_is_integer_1_to_32() {
        for value in [1, 2, 7, 31, 32] {
            assert!(
                OCCUPANCY_DOMAIN.accepts(value),
                "occupancy {value} must be accepted"
            );
        }
        for value in [i32::MIN, -1, 0, 33, 64, i32::MAX] {
            assert!(
                !OCCUPANCY_DOMAIN.accepts(value),
                "occupancy {value} must be rejected"
            );
        }
    }

    #[test]
    fn num_cta_in_cga_domain_is_power_of_two_1_to_16() {
        for value in [1, 2, 4, 8, 16] {
            assert!(
                NUM_CTA_IN_CGA_DOMAIN.accepts(value),
                "num_cta_in_cga {value} must be accepted"
            );
        }
        for value in [-1, 0, 3, 5, 6, 7, 12, 15, 17, 32] {
            assert!(
                !NUM_CTA_IN_CGA_DOMAIN.accepts(value),
                "num_cta_in_cga {value} must be rejected"
            );
        }
    }

    #[test]
    fn num_worker_warps_domain_is_power_of_two_1_to_32() {
        for value in [1, 2, 4, 8, 16, 32] {
            assert!(
                NUM_WORKER_WARPS_PER_CTA_DOMAIN.accepts(value),
                "num_worker_warps_per_cta {value} must be accepted"
            );
        }
        for value in [-1, 0, 3, 6, 12, 24, 33, 64] {
            assert!(
                !NUM_WORKER_WARPS_PER_CTA_DOMAIN.accepts(value),
                "num_worker_warps_per_cta {value} must be rejected"
            );
        }
    }

    #[test]
    fn latency_domain_is_integer_1_to_10() {
        for value in [1, 2, 5, 10] {
            assert!(
                LATENCY_DOMAIN.accepts(value),
                "latency {value} must be accepted"
            );
        }
        for value in [-5, -1, 0, 11, 100] {
            assert!(
                !LATENCY_DOMAIN.accepts(value),
                "latency {value} must be rejected"
            );
        }
    }

    #[test]
    fn negative_values_are_rejected_without_unsigned_wraparound() {
        // A value that would become a huge positive number in an unsigned
        // reinterpretation must still be reported as itself.
        let err = check_hint_value("occupancy", -1, OCCUPANCY_DOMAIN)
            .expect_err("negative occupancy must be rejected");
        assert_eq!(err.value, -1);
        assert!(
            err.message().contains("-1"),
            "message must quote the real value, got: {}",
            err.message()
        );
    }

    #[test]
    fn error_message_states_the_valid_range() {
        let err =
            check_hint_value("occupancy", 33, OCCUPANCY_DOMAIN).expect_err("33 must be rejected");
        let message = err.message();
        assert!(
            message.contains("occupancy"),
            "message must name the hint: {message}"
        );
        assert!(
            message.contains("[1, 32]"),
            "message must state the range: {message}"
        );

        let err = check_hint_value("num_cta_in_cga", 3, NUM_CTA_IN_CGA_DOMAIN)
            .expect_err("3 must be rejected");
        assert!(
            err.message().contains("power of two"),
            "power-of-two hints must say so: {}",
            err.message()
        );
    }

    // -- Macro path -------------------------------------------------------

    #[test]
    fn macro_path_accepts_boundary_values() {
        let expr = hints_expr("occupancy = 32, num_cta_in_cga = 16, num_worker_warps_per_cta = 32");
        let hints = parse_hints(&expr, &NoLocation).expect("boundary values are legal");
        let sm = hints.get_sm_hints("sm_89").expect("sm_89 entry exists");
        assert_eq!(sm.occupancy, Some(32));
        assert_eq!(sm.num_cta_in_cga, Some(16));
        assert_eq!(sm.num_worker_warps_per_cta, Some(32));
    }

    #[test]
    fn macro_path_rejects_occupancy_out_of_range_with_a_location() {
        let expr: Expr = hints_expr("occupancy = 33");
        let err = hints_error(parse_hints(&expr, &FixedLocation), "33 must be rejected");
        match err {
            JITError::Located(message, location) => {
                assert!(message.contains("occupancy"), "message: {message}");
                assert!(message.contains("[1, 32]"), "message: {message}");
                assert_eq!(location.line, 7);
                assert_eq!(location.file, "kernel.rs");
            }
            other => panic!("expected a located error, got {other:?}"),
        }
    }

    #[test]
    fn macro_path_rejects_non_power_of_two_cga_size() {
        let expr: Expr = hints_expr("num_cta_in_cga = 3");
        let err = hints_error(parse_hints(&expr, &FixedLocation), "3 must be rejected");
        assert!(
            error_text(&err).contains("num_cta_in_cga"),
            "message: {}",
            error_text(&err)
        );
    }

    #[test]
    fn macro_path_rejects_worker_warps_above_32() {
        let expr: Expr = hints_expr("num_worker_warps_per_cta = 33");
        let err = hints_error(parse_hints(&expr, &FixedLocation), "33 must be rejected");
        assert!(
            error_text(&err).contains("num_worker_warps_per_cta"),
            "message: {}",
            error_text(&err)
        );
    }

    #[test]
    fn macro_path_rejects_zero_and_negative() {
        for value in [-1i32, 0] {
            let expr = hints_expr(&format!("occupancy = {value}"));
            assert!(
                parse_hints(&expr, &FixedLocation).is_err(),
                "occupancy {value} must be rejected"
            );

            let expr = hints_expr(&format!("num_cta_in_cga = {value}"));
            assert!(
                parse_hints(&expr, &FixedLocation).is_err(),
                "num_cta_in_cga {value} must be rejected"
            );
        }
    }

    #[test]
    fn macro_path_still_rejects_per_op_hints_at_entry_level() {
        let expr: Expr = hints_expr("latency = 4");
        let err = hints_error(parse_hints(&expr, &NoLocation), "latency is per-op");
        assert!(
            error_text(&err).contains("per-op"),
            "message: {}",
            error_text(&err)
        );
    }

    #[test]
    fn macro_path_still_rejects_duplicate_keys() {
        // A duplicate key cannot be expressed through `hints_expr`, which
        // builds the value for one architecture. Spell the full attribute
        // value so the duplicate sits where the parser will see it.
        let expr: Expr = syn::parse_str("(sm_89 = (occupancy = 4, occupancy = 8,),)")
            .expect("duplicate-key tuple parses");
        let err = hints_error(parse_hints(&expr, &NoLocation), "duplicate key");
        assert!(
            error_text(&err).contains("already been set"),
            "message: {}",
            error_text(&err)
        );
    }

    #[test]
    fn macro_path_still_requires_integer_literals() {
        let expr: Expr = hints_expr("occupancy = 4.5");
        assert!(parse_hints(&expr, &NoLocation).is_err());

        let expr: Expr = hints_expr("occupancy = \"8\"");
        assert!(parse_hints(&expr, &NoLocation).is_err());
    }

    #[test]
    fn unlocated_diagnostics_stay_generic() {
        // A synthetic AST has no `SpanBase`. The diagnostic must not pretend
        // to know a position.
        let expr: Expr = hints_expr("occupancy = 99");
        let err = hints_error(parse_hints(&expr, &NoLocation), "99 must be rejected");
        assert!(
            matches!(err, JITError::Generic(_)),
            "expected Generic without a source anchor, got {err:?}"
        );
    }

    // -- Runtime path -----------------------------------------------------

    #[test]
    fn builder_accepts_boundary_values() {
        let opts = CompileOptions::default()
            .occupancy(1)
            .expect("1 is legal")
            .num_cta_in_cga(16)
            .expect("16 is legal")
            .num_worker_warps_per_cta(32)
            .expect("32 is legal");
        assert_eq!(opts.occupancy, Some(1));
        assert_eq!(opts.num_cta_in_cga, Some(16));
        assert_eq!(opts.num_worker_warps_per_cta, Some(32));
        assert!(opts.validate().is_ok());
    }

    #[test]
    fn builder_rejects_out_of_range_values() {
        assert!(CompileOptions::default().occupancy(0).is_err());
        assert!(CompileOptions::default().occupancy(33).is_err());
        assert!(CompileOptions::default().num_cta_in_cga(3).is_err());
        assert!(CompileOptions::default().num_cta_in_cga(32).is_err());
        assert!(CompileOptions::default()
            .num_worker_warps_per_cta(6)
            .is_err());
        assert!(CompileOptions::default()
            .num_worker_warps_per_cta(64)
            .is_err());
    }

    #[test]
    fn max_divisibility_is_accepted_without_a_range_check() {
        for value in [-1, 0, 1, 4, 16, 4096] {
            assert!(
                CompileOptions::default().max_divisibility(value).is_ok(),
                "max_divisibility {value} must stay accepted"
            );
        }
    }

    #[test]
    fn direct_field_writes_are_caught_by_validate() {
        // The hint fields are public, so this is exactly the path the builder
        // cannot protect. `validate` must catch it.
        let mut opts = CompileOptions::default();
        opts.occupancy = Some(99);
        let err = opts.validate().expect_err("99 must be rejected");
        assert_eq!(err.hint, "occupancy");
        assert_eq!(err.value, 99);

        let mut opts = CompileOptions::default();
        opts.num_cta_in_cga = Some(12);
        assert!(opts.validate().is_err());

        let mut opts = CompileOptions::default();
        opts.num_worker_warps_per_cta = Some(3);
        assert!(opts.validate().is_err());
    }

    #[test]
    fn validate_accepts_default_and_empty_options() {
        assert!(CompileOptions::default().validate().is_ok());
        assert!(CompileOptions::new().validate().is_ok());
    }

    #[test]
    fn apply_compile_options_rejects_an_invalid_direct_field_write() {
        let mut hints = OptimizationHints::empty();
        hints.target_gpu_name = Some("sm_89".to_string());
        let mut options = CompileOptions::default();
        options.occupancy = Some(0);
        assert!(
            hints.apply_compile_options(&options).is_err(),
            "a directly written invalid value must not be applied"
        );
        assert!(
            hints.get_sm_hints("sm_89").is_none(),
            "nothing may be recorded for a rejected option set"
        );
    }

    #[test]
    fn apply_compile_options_still_overrides_entry_hints() {
        let expr: Expr = hints_expr("occupancy = 4, num_cta_in_cga = 2");
        let mut hints = parse_hints(&expr, &NoLocation).expect("entry hints parse");
        let options = CompileOptions::default()
            .occupancy(8)
            .expect("legal")
            .num_cta_in_cga(4)
            .expect("legal");
        hints
            .apply_compile_options(&options)
            .expect("valid options apply");
        let sm = hints.get_sm_hints("sm_89").expect("sm_89 entry exists");
        assert_eq!(sm.occupancy, Some(8), "runtime must override entry hint");
        assert_eq!(sm.num_cta_in_cga, Some(4));
    }
}
