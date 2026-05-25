/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Common test utilities and constants shared across all test modules.

use std::sync::{Mutex, MutexGuard, OnceLock};

/// Process-wide lock serializing tests that assert on global kernel-cache
/// state (presence of a key) or the global JIT compile counter.
///
/// The in-memory kernel cache and `cutile::tile_kernel::jit_compile_count()`
/// are process-global. A test that measures "did this call compile or hit the
/// cache" must hold this lock for the whole measured window so no other test's
/// concurrent compile can move the counter between snapshots. All cache-state
/// tests (in `warmup.rs` and `warmup_bench.rs`) share this single lock.
#[allow(dead_code)]
pub fn cache_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Stack size for test threads.
///
/// Tests require larger stack sizes due to:
/// - Deep MLIR AST structures during compilation
/// - Multiple unary operations in single test kernels
/// - Nested function calls in the compiler
///
/// Binary search determined minimum requirements:
/// - Basic tests: ~2.121 MB
/// - With assume variants: ~2.612 MB
/// - With reduce/scan operations: ~2.7 MB
/// - With all unary math operations: ~5 MB (after adding absf, negf, negi, floor)
/// - tensor_views module tests require a bit more headroom.
/// Using 8 MB provides an adequate safety margin for all tests.
pub const TEST_STACK_SIZE: usize = 8_000_000; // 8 MB

/// Helper to run a test with the required stack size.
///
/// # Example
///
/// ```rust,ignore
/// #[test]
/// fn my_test() {
///     common::with_test_stack(|| {
///         // Your test code here
///     });
/// }
/// ```
pub fn with_test_stack<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    std::thread::Builder::new()
        .stack_size(TEST_STACK_SIZE)
        .spawn(f)
        .expect("Failed to spawn test thread")
        .join()
        .expect("Test thread panicked")
}
