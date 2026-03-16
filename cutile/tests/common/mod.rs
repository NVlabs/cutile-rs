/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/// Common test utilities and constants shared across all test modules.

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
/// Using 5 MB provides adequate safety margin for all tests.
pub const TEST_STACK_SIZE: usize = 5_000_000; // 5 MB

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
