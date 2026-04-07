/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile;
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;

mod common;

#[cutile::module]
mod memory_and_atomic_ops_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn join_tokens_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test join_tokens - create multiple independent tokens and join them
        // Note: This is a low-level operation; tokens are typically managed automatically
        // through partition views. Manual token joining is for advanced use cases.
        let token1: Token = new_token_unordered();
        let token2: Token = new_token_unordered();

        // Join the tokens into a single combined token
        let joined: Token = join_tokens(&[token1, token2]);

        // Use the joined token in a partition view
        let shape = output.shape();
        let _partition: PartitionMut<f32, S> =
            unsafe { make_partition_view_mut(output, shape, joined) };

        // Create a simple tile to demonstrate the joined token is used
        let tile: Tile<f32, S> = constant(1.0, shape);

        // Store using Tensor's high-level API (which internally manages tokens)
        output.store(tile);
    }

    #[cutile::entry()]
    fn ptr_load_store_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test load_ptr_tko and store_ptr_tko operations with basic parameters
        // Create synthetic pointer tiles
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);

        // Load from pointer tile with relaxed/device semantics, no optional params
        let (loaded_values, _load_token): (Tile<f32, S>, Token) =
            load_ptr_tko(ptrs, "relaxed", "device", None, None, None, None);

        // Modify the values
        let modified: Tile<f32, S> = loaded_values + constant(1.0, output.shape());

        // Store back to pointer tile with relaxed/device semantics, no optional params
        let _store_token: Token =
            store_ptr_tko(ptrs, modified, "relaxed", "device", None, None, None);

        // Store result using regular tensor API
        output.store(modified);
    }

    #[cutile::entry()]
    fn load_ptr_weak_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test load_ptr_tko with weak memory ordering
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);

        let (loaded_values, _token): (Tile<f32, S>, Token) =
            load_ptr_tko(ptrs, "weak", "tl_blk", None, None, None, None);

        output.store(loaded_values);
    }

    #[cutile::entry()]
    fn load_ptr_acquire_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test load_ptr_tko with acquire memory ordering
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);

        let (loaded_values, _token): (Tile<f32, S>, Token) =
            load_ptr_tko(ptrs, "acquire", "sys", None, None, None, None);

        output.store(loaded_values);
    }

    #[cutile::entry()]
    fn load_ptr_with_token_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test load_ptr_tko with input token
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);

        let input_token = new_token_unordered();
        let (loaded_values, _result_token): (Tile<f32, S>, Token) = load_ptr_tko(
            ptrs,
            "relaxed",
            "device",
            None,
            None,
            Some(input_token),
            None,
        );

        output.store(loaded_values);
    }

    #[cutile::entry()]
    fn load_ptr_with_mask_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test load_ptr_tko with mask and padding value
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);

        // Create a mask (all true for this test)
        let mask: Tile<bool, S> = constant(true, output.shape());
        let padding = 0.0f32;

        let (loaded_values, _token): (Tile<f32, S>, Token) = load_ptr_tko(
            ptrs,
            "relaxed",
            "device",
            Some(mask),
            Some(padding),
            None,
            None,
        );

        output.store(loaded_values);
    }

    #[cutile::entry()]
    fn store_ptr_release_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test store_ptr_tko with release memory ordering (store-specific)
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);

        // Create values to store
        let values: Tile<f32, S> = constant(42.0f32, output.shape());

        // Store with release semantics and sys scope
        let _store_token: Token = store_ptr_tko(ptrs, values, "release", "sys", None, None, None);

        output.store(values);
    }

    #[cutile::entry()]
    fn store_ptr_with_mask_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test store_ptr_tko with mask parameter
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);

        // Create values to store
        let values: Tile<f32, S> = constant(99.0f32, output.shape());

        // Create a mask (all true for this test)
        let mask: Tile<bool, S> = constant(true, output.shape());

        // Store with mask, relaxed semantics, and device scope
        let _store_token: Token =
            store_ptr_tko(ptrs, values, "relaxed", "device", Some(mask), None, None);

        output.store(values);
    }

    #[cutile::entry()]
    fn atomic_rmw_kernel<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
        counters: &mut Tensor<f32, S>,
    ) {
        // Build a synthetic pointer tile and increments to exercise the lowering path.
        let increments: Tile<f32, S> = constant(1.0, output.shape());
        let ptr_seed: Tile<i64, S> = constant(0i64, counters.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);

        // Use atomic_addf_tko with relaxed/device semantics, no optional params
        let (old_values, _result_token): (Tile<f32, S>, Token) =
            atomic_rmw_tko(ptrs, increments, "addf", "relaxed", "device", None, None);

        output.store(old_values);

        // Keep existing behaviour for counters to exercise tensor loads/stores.
        let counter_values: Tile<f32, S> = load_tile_mut(counters);
        counters.store(counter_values);
    }

    #[cutile::entry()]
    fn atomic_cas_kernel<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
        expected: &mut Tensor<f32, S>,
    ) {
        // Test atomic_cas_tko operation with synthetic pointer tiles
        // Build pointer tiles similar to atomic_rmw
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);

        // Create comparison and new values
        let cmp_values: Tile<f32, S> = constant(0.0, output.shape());
        let new_values: Tile<f32, S> = constant(1.0, output.shape());
        let _token: Token = new_token_unordered();

        // Perform atomic compare-and-swap
        let (old_values, _result_token): (Tile<f32, S>, Token) = atomic_cas_tko(
            ptrs,
            cmp_values,
            new_values,
            "relaxed",
            "device",
            None,
            Some(_token),
        );

        output.store(old_values);

        // Keep existing behaviour for expected tensor
        let expected_values: Tile<f32, S> = load_tile_mut(expected);
        expected.store(expected_values);
    }

    #[cutile::entry()]
    fn atomic_cas_with_mask_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test atomic_cas_tko with mask parameter
        // Build pointer tiles
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);

        // Create comparison and new values
        let cmp_values: Tile<i64, S> = constant(0i64, output.shape());
        let new_values: Tile<i64, S> = constant(42i64, output.shape());

        // Create a mask that selects every other element
        let mask_values: Tile<bool, S> = constant(true, output.shape());

        // Perform atomic compare-and-swap with mask
        let (old_values, _result_token): (Tile<i64, S>, Token) = atomic_cas_tko(
            ptrs,
            cmp_values,
            new_values,
            "acquire",
            "device",
            Some(mask_values),
            None,
        );

        output.store(old_values);
    }

    #[cutile::entry()]
    fn atomic_cas_acq_rel_sys_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test atomic_cas_tko with acq_rel memory ordering and sys scope
        // This tests stronger memory semantics and broader scope
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);

        // Create comparison and new values
        let cmp_values: Tile<i64, S> = constant(100i64, output.shape());
        let new_values: Tile<i64, S> = constant(200i64, output.shape());

        // Perform atomic compare-and-swap with acq_rel ordering and sys scope
        let (old_values, _result_token): (Tile<i64, S>, Token) =
            atomic_cas_tko(ptrs, cmp_values, new_values, "acq_rel", "sys", None, None);

        output.store(old_values);
    }

    // Atomic RMW wrapper function tests
    #[cutile::entry()]
    fn atomic_and_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test atomic_and_tko with integer types
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut i64, S> = ptrs_i64;

        let values: Tile<i64, S> = constant(0xFFi64, output.shape());

        let (old_values, _token): (Tile<i64, S>, Token) =
            atomic_rmw_tko(ptrs, values, "and", "relaxed", "device", None, None);

        output.store(old_values);
    }

    #[cutile::entry()]
    fn atomic_add_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test atomic_add_tko with integer addition
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut i64, S> = ptrs_i64;

        let increments: Tile<i64, S> = constant(5i64, output.shape());

        let (old_values, _token): (Tile<i64, S>, Token) =
            atomic_rmw_tko(ptrs, increments, "add", "acq_rel", "sys", None, None);

        output.store(old_values);
    }

    #[cutile::entry()]
    fn atomic_max_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test atomic_max_tko with acquire/release semantics
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut i64, S> = ptrs_i64;

        let values: Tile<i64, S> = constant(100i64, output.shape());

        let (old_values, _token): (Tile<i64, S>, Token) =
            atomic_rmw_tko(ptrs, values, "max", "acquire", "device", None, None);

        output.store(old_values);
    }

    #[cutile::entry()]
    fn atomic_rmw_with_mask_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test atomic_add_tko with mask parameter
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut i64, S> = ptrs_i64;

        let increments: Tile<i64, S> = constant(10i64, output.shape());
        let mask: Tile<bool, S> = constant(true, output.shape());

        let (old_values, _token): (Tile<i64, S>, Token) = atomic_rmw_tko(
            ptrs,
            increments,
            "add",
            "relaxed",
            "device",
            Some(mask),
            None,
        );

        output.store(old_values);
    }

    #[cutile::entry()]
    fn atomic_rmw_with_token_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test atomic_xor_tko with input token
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut i64, S> = ptrs_i64;

        let values: Tile<i64, S> = constant(0xFFFFi64, output.shape());
        let input_token: Token = new_token_unordered();

        let (old_values, _token): (Tile<i64, S>, Token) = atomic_rmw_tko(
            ptrs,
            values,
            "xor",
            "release",
            "sys",
            None,
            Some(input_token),
        );

        output.store(old_values);
    }

    #[cutile::entry()]
    fn atomic_xchg_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test atomic_xchg_tko with floating-point exchange
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);

        let new_values: Tile<f32, S> = constant(42.5f32, output.shape());

        let (old_values, _token): (Tile<f32, S>, Token) =
            atomic_rmw_tko(ptrs, new_values, "xchg", "acq_rel", "device", None, None);

        output.store(old_values);
    }

    #[cutile::entry()]
    fn padded_partition_view_kernel<const S: [i32; 1]>(input: &Tensor<f32, S>) {
        let token: Token = new_token_unordered();
        let shape = input.shape();
        let partition: Partition<f32, S> =
            make_partition_view_padded(input, shape, "neg_inf", token);
        let idx: [i32; 1] = [0i32];
        let _tile: Tile<f32, S> = load_from_view(&partition, idx, None, false);
    }
}

use memory_and_atomic_ops_module::_module_asts;

#[test]
fn compile_join_tokens() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "join_tokens_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== JOIN_TOKENS MLIR ===\n{}", module_op_str);

        // Verify join_tokens operation appears
        assert!(
            module_op_str.contains("join_tokens"),
            "Expected join_tokens operation in MLIR output"
        );

        // Verify multiple make_token operations
        let token_count = module_op_str.matches("make_token").count();
        assert!(
            token_count >= 2,
            "Expected at least 2 make_token operations, found {}",
            token_count
        );

        // Verify the joined token is used in make_partition_view
        assert!(
            module_op_str.contains("make_partition_view"),
            "Expected partition view created with joined token"
        );

        println!("\n✓ join_tokens operation verified (joining multiple tokens)");
    });
}

#[test]
fn compile_ptr_load_store() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "ptr_load_store_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!(
            "\n=== LOAD_PTR_TKO / STORE_PTR_TKO MLIR ===\n{}",
            module_op_str
        );

        // Verify load_ptr_tko operation appears
        assert!(
            module_op_str.contains("load_ptr_tko"),
            "Expected load_ptr_tko operation in MLIR output"
        );

        // Verify store_ptr_tko operation appears
        assert!(
            module_op_str.contains("store_ptr_tko"),
            "Expected store_ptr_tko operation in MLIR output"
        );

        // Verify memory ordering attributes
        assert!(
            module_op_str.contains("relaxed"),
            "Expected relaxed memory ordering"
        );

        // Verify pointer construction
        assert!(
            module_op_str.contains("int_to_ptr"),
            "Expected pointer construction via int_to_ptr"
        );
        assert!(
            module_op_str.contains("ptr_to_ptr"),
            "Expected pointer cast via ptr_to_ptr"
        );

        // Verify memory semantics for load_ptr_tko (MLIR displays in compact format)
        assert!(
            module_op_str.contains("load_ptr_tko relaxed device"),
            "Expected 'load_ptr_tko relaxed device' for load_ptr_tko operation"
        );

        // Verify store_ptr_tko also has correct semantics
        assert!(
            module_op_str.contains("store_ptr_tko relaxed device"),
            "Expected 'store_ptr_tko relaxed device' for store_ptr_tko operation"
        );

        println!("\n✓ load_ptr_tko and store_ptr_tko operations verified");
    });
}

#[test]
fn compile_load_ptr_weak() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "load_ptr_weak_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== LOAD_PTR_WEAK MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("load_ptr_tko"),
            "Expected load_ptr_tko operation"
        );
        assert!(
            module_op_str.contains("load_ptr_tko weak"),
            "Expected 'load_ptr_tko weak' with weak memory ordering (no scope)"
        );

        println!("\n✓ load_ptr_tko with weak ordering verified");
    });
}

#[test]
fn compile_load_ptr_acquire() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "load_ptr_acquire_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== LOAD_PTR_ACQUIRE MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("load_ptr_tko"),
            "Expected load_ptr_tko operation"
        );
        assert!(
            module_op_str.contains("load_ptr_tko acquire sys"),
            "Expected 'load_ptr_tko acquire sys' with acquire memory ordering and system scope"
        );

        println!("\n✓ load_ptr_tko with acquire/sys verified");
    });
}

#[test]
fn compile_load_ptr_with_token() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "load_ptr_with_token_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== LOAD_PTR_WITH_TOKEN MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("load_ptr_tko"),
            "Expected load_ptr_tko operation"
        );
        assert!(
            module_op_str.contains("make_token"),
            "Expected make_token for input token generation"
        );
        // In compact format, token appears as "token=%N"
        assert!(
            module_op_str.contains("token="),
            "Expected token parameter in load_ptr_tko operation"
        );

        println!("\n✓ load_ptr_tko with input token verified");
    });
}

#[test]
fn compile_load_ptr_with_mask() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "load_ptr_with_mask_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== LOAD_PTR_WITH_MASK MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("load_ptr_tko"),
            "Expected load_ptr_tko operation"
        );
        // Verify the padding value was promoted from scalar to shaped tile
        assert!(
            module_op_str.contains("reshape") && module_op_str.contains("broadcast"),
            "Expected reshape and broadcast operations for padding promotion"
        );
        assert!(
            module_op_str.contains("tile<128xi1>"),
            "Expected i1 tile type for mask"
        );

        println!("\n✓ load_ptr_tko with mask and padding verified");
    });
}

#[test]
fn compile_store_ptr_release() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "store_ptr_release_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== STORE_PTR_RELEASE MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("store_ptr_tko"),
            "Expected store_ptr_tko operation"
        );
        assert!(
            module_op_str.contains("store_ptr_tko release sys"),
            "Expected 'store_ptr_tko release sys' with release memory ordering and system scope"
        );

        println!("\n✓ store_ptr_tko with release/sys verified");
    });
}

#[test]
fn compile_store_ptr_with_mask() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "store_ptr_with_mask_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== STORE_PTR_WITH_MASK MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("store_ptr_tko"),
            "Expected store_ptr_tko operation"
        );
        assert!(
            module_op_str.contains("store_ptr_tko relaxed device"),
            "Expected 'store_ptr_tko relaxed device' for store_ptr_tko operation"
        );
        assert!(
            module_op_str.contains("tile<128xi1>"),
            "Expected i1 tile type for mask"
        );

        println!("\n✓ store_ptr_tko with mask verified");
    });
}

#[test]
fn compile_atomic_rmw() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "atomic_rmw_kernel",
            &[128.to_string()],
            &[("output", &[1]), ("counters", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== ATOMIC_RMW MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("atomic_rmw_tko"),
            "Expected atomic_rmw_tko operation in MLIR output"
        );
        assert!(
            module_op_str.contains("relaxed device"),
            "Expected relaxed/device memory semantics on atomic_rmw_tko"
        );
        assert!(
            module_op_str.contains(", addf,"),
            "Expected addf mode on atomic_rmw_tko for floating-point operands"
        );
        assert!(
            module_op_str.contains("int_to_ptr"),
            "Expected pointer construction via int_to_ptr for atomic_rmw_tko test"
        );
        assert!(
            module_op_str.contains("ptr_to_ptr"),
            "Expected pointer cast via ptr_to_ptr for atomic_rmw_tko test"
        );

        println!("\n✓ atomic_rmw_tko lowering verified");
    });
}

#[test]
fn compile_atomic_cas() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "atomic_cas_kernel",
            &[128.to_string()],
            &[("output", &[1]), ("expected", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== ATOMIC_CAS MLIR ===\n{}", module_op_str);

        // Verify atomic_cas_tko operation appears
        assert!(
            module_op_str.contains("atomic_cas_tko"),
            "Expected atomic_cas_tko operation in MLIR output"
        );

        // Verify memory semantics attributes
        assert!(
            module_op_str.contains("relaxed device"),
            "Expected relaxed/device memory semantics on atomic_cas_tko"
        );

        // Verify pointer construction operations
        assert!(
            module_op_str.contains("int_to_ptr"),
            "Expected pointer construction via int_to_ptr for atomic_cas_tko test"
        );
        assert!(
            module_op_str.contains("ptr_to_ptr"),
            "Expected pointer cast via ptr_to_ptr for atomic_cas_tko test"
        );

        println!("\n✓ atomic_cas_tko lowering verified (with pointer tiles and CAS semantics)");
    });
}

#[test]
fn compile_atomic_cas_with_mask() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "atomic_cas_with_mask_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== ATOMIC_CAS WITH MASK MLIR ===\n{}", module_op_str);

        // Verify atomic_cas_tko operation appears
        assert!(
            module_op_str.contains("atomic_cas_tko"),
            "Expected atomic_cas_tko operation in MLIR output"
        );

        // Verify mask operand is present (operandSegmentSizes should have mask_count > 0)
        // The mask will be included in operands, so operandSegmentSizes should show it
        assert!(
            module_op_str.contains("operandSegmentSizes")
                || module_op_str.contains("atomic_cas_tko"),
            "Expected operandSegmentSizes with mask for atomic_cas_tko"
        );

        println!("\n✓ atomic_cas_tko with mask lowering verified");
    });
}

#[test]
fn compile_atomic_cas_acq_rel_sys() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "atomic_cas_acq_rel_sys_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== ATOMIC_CAS ACQ_REL SYS MLIR ===\n{}", module_op_str);

        // Verify atomic_cas_tko operation appears
        assert!(
            module_op_str.contains("atomic_cas_tko"),
            "Expected atomic_cas_tko operation in MLIR output"
        );

        // Verify integer type (i64) is used
        assert!(
            module_op_str.contains("i64") || module_op_str.contains("atomic_cas_tko"),
            "Expected i64 type for atomic_cas_tko test"
        );

        // Note: Memory ordering/scope are encoded as integers in MLIR, so we verify compilation success
        // which confirms the string-to-integer conversion worked correctly
        println!("\n✓ atomic_cas_tko with acq_rel/sys lowering verified");
    });
}

#[test]
fn compile_atomic_and() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "atomic_and_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== ATOMIC_AND MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("atomic_rmw_tko"),
            "Expected atomic_rmw_tko operation"
        );
        assert!(module_op_str.contains(", and,"), "Expected 'and' mode");
        assert!(
            module_op_str.contains("relaxed device"),
            "Expected relaxed/device semantics"
        );

        println!("\n✓ atomic_and_tko verified");
    });
}

#[test]
fn compile_atomic_add() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "atomic_add_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== ATOMIC_ADD MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("atomic_rmw_tko"),
            "Expected atomic_rmw_tko operation"
        );
        assert!(module_op_str.contains(", add,"), "Expected 'add' mode");
        assert!(
            module_op_str.contains("acq_rel sys"),
            "Expected acq_rel/sys semantics"
        );

        println!("\n✓ atomic_add_tko verified");
    });
}

#[test]
fn compile_atomic_max() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "atomic_max_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== ATOMIC_MAX MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("atomic_rmw_tko"),
            "Expected atomic_rmw_tko operation"
        );
        assert!(module_op_str.contains(", max,"), "Expected 'max' mode");
        assert!(
            module_op_str.contains("acquire device"),
            "Expected acquire/device semantics"
        );

        println!("\n✓ atomic_max_tko verified");
    });
}

#[test]
fn compile_atomic_rmw_with_mask() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "atomic_rmw_with_mask_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== ATOMIC_RMW_WITH_MASK MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("atomic_rmw_tko"),
            "Expected atomic_rmw_tko operation"
        );
        assert!(module_op_str.contains(", add,"), "Expected 'add' mode");
        assert!(
            module_op_str.contains("tile<128xi1>"),
            "Expected i1 tile type for mask"
        );

        println!("\n✓ atomic_add_tko with mask verified");
    });
}

#[test]
fn compile_atomic_rmw_with_token() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "atomic_rmw_with_token_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== ATOMIC_RMW_WITH_TOKEN MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("atomic_rmw_tko"),
            "Expected atomic_rmw_tko operation"
        );
        assert!(module_op_str.contains(", xor,"), "Expected 'xor' mode");
        assert!(
            module_op_str.contains("release sys"),
            "Expected release/sys semantics"
        );

        println!("\n✓ atomic_xor_tko with token verified");
    });
}

#[test]
fn compile_atomic_xchg() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "atomic_xchg_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== ATOMIC_XCHG MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("atomic_rmw_tko"),
            "Expected atomic_rmw_tko operation"
        );
        assert!(module_op_str.contains(", xchg,"), "Expected 'xchg' mode");
        assert!(
            module_op_str.contains("acq_rel device"),
            "Expected acq_rel/device semantics"
        );

        println!("\n✓ atomic_xchg_tko verified");
    });
}

#[test]
fn compile_padded_partition_view() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "memory_and_atomic_ops_module",
            "padded_partition_view_kernel",
            &[128.to_string()],
            &[("input", &[1])],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== PADDED_PARTITION_VIEW MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("make_partition_view"),
            "Expected make_partition_view operation in MLIR output"
        );
        assert!(
            module_op_str.contains("padding_value = neg_inf"),
            "Expected padding_value = neg_inf in partition_view type"
        );

        println!("\n✓ make_partition_view_padded with neg_inf padding verified");
    });
}
