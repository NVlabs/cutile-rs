/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! AST pass pipeline for the cuTile compiler.
//!
//! Passes transform or annotate the syn AST in a defined order before IR
//! emission. Each pass reads the previous pass's output and produces a
//! progressively more resolved form.
//!
//! ```text
//! Raw syn AST (from proc macro)
//!     ↓
//! [Pass 1: Name Resolution]  — build symbol table, resolve imports
//!     ↓
//! [Pass 2: Type Inference]   — (future) every expression typed
//!     ↓
//! [Pass 3: Instantiation]    — (future) no generics remain
//!     ↓
//! [IR Emission]              — translation to cutile-ir
//! ```

pub mod name_resolution;
pub mod node_ids;
pub mod type_inference;
pub mod typed_dispatch_lowering;
