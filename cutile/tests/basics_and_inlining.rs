/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;

mod common;

#[cutile::module]
mod basics_and_inlining_module {

    #![allow(unused_variables)]

    use cutile::core::*;

    // Inlining

    fn other_function<T: ElementType, const D: [i32; 3], const B: [i32; 3]>(
        y: Tile<T, D>,
        shape: Shape<B>,
    ) -> Tile<T, B> {
        reshape(y, shape)
    }

    #[cutile::entry()]
    fn inlining_kernel<E: ElementType, const X: i32, const S: [i32; 3], const Y: i32>(
        x: f32,
        y: &mut Tensor<E, S>,
    ) {
        let tile_x: Tile<f32, S> = broadcast_scalar(x, y.shape());
        let empty: &[i32] = &[];
        let shape: Shape<{ [32, 512, 1024] }> = Shape::<{ [32, 512, 1024] }> { dims: empty };
        other_function(tile_x, shape);
    }

    // Various Rust->TileIR tests.

    pub struct SomeStruct {
        pub ptr: *mut f32,
        pub scalar: f32,
    }

    fn identity(x: *mut f32) -> *mut f32 {
        return x;
    }

    fn ones_shape<const N: usize>() -> [i32; N] {
        [1i32; N]
    }

    #[cutile::entry()]
    unsafe fn basics_kernel<const S: [i32; 3]>(
        y: &mut Tensor<f32, { [128, -1] }>,
        #[allow(unused_variables)] w: &Tensor<f32, S>,
        ptr: *mut f32,
        scalar: f32,
        integer: u32,
    ) {
        let some_struct: SomeStruct = SomeStruct { ptr, scalar };
        let the_ptr = some_struct.ptr;
        let mut _result = identity(the_ptr);
        _result = the_ptr;

        let tile_scalar: Tile<f32, { [] }> = scalar_to_tile(scalar);
        let _scalar2: f32 = tile_to_scalar(tile_scalar);

        let tile_integer: Tile<u32, { [] }> = scalar_to_tile(integer);
        let _integer2: u32 = tile_to_scalar(tile_integer);

        let ptr_tile: PointerTile<*mut f32, { [] }> = pointer_to_tile(ptr);
        let _ptr2: *mut f32 = tile_to_pointer(ptr_tile);

        let num_pid: (i32, i32, i32) = get_num_tile_blocks();
        let shape: Shape<{ [128, 256] }> = Shape::<{ [128, 256] }> {
            dims: &[num_pid.0, 256i32],
        };

        let shape_dim_1: i32 = 128;
        let shape_dim_2: i32 = 256;
        let stride_dim_1: i32 = num_pid.0;
        let stride: Array<{ [-1, 128] }> = Array::<{ [-1, 128] }> {
            dims: &[stride_dim_1],
        };
        let dynamic_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> {
            dims: &[shape_dim_1, shape_dim_2],
        };

        unsafe {
            let token: Token = new_token_unordered();
            let _some_tensor: Tensor<f32, { [-1, -1] }> =
                make_tensor_view(ptr_tile, dynamic_shape, stride, token);
            let mut partition: PartitionMut<f32, { [128, 256] }> =
                make_partition_view_mut(&y, shape, token);
            let idx: [i32; 2] = [0i32, 0i32];
            let some_tile: Tile<f32, { [128, 256] }> = load_from_view_mut(&partition, idx);
            store_to_view_mut(&mut partition, some_tile, idx);
            let _store_token_2: Token = store_to_view_mut(&mut partition, some_tile, idx);
        }

        let shape: Shape<{ [1, 1] }> = Shape::<{ [1, 1] }> {
            dims: &[1i32, 1i32],
        };
        let x: Tile<u32, { [1, 1] }> = reshape(tile_integer, shape);
        let x_shape: Shape<{ [128, 64] }> = Shape::<{ [128, 64] }> {
            dims: &[128i32, 64i32],
        };
        let x_shaped: Tile<u32, { [128, 64] }> = broadcast(x, x_shape);
        let trans_perm: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let _x_transpose: Tile<u32, { [64, 128] }> = permute(x_shaped, trans_perm);
        let _this_works: i32 = 1i32 + 2i32;

        let _multi_dim_const: Tile<f32, { [128, 64] }> = constant(0.0, x_shape);
        let _neg_const: Tile<f32, { [128, 64] }> = constant(-1.0, x_shape);

        // Test tuples.
        let tuple: (i32, i32) = (1i32, 2i32);
        let _tuple_0: i32 = tuple.0;

        let _ones: [i32; 2] = ones_shape::<2>();
        let _a_bool: bool = false;

        let xu32: u32 = 1;
        let yu32: u32 = 2;
        let _zu32 = xu32 / yu32;

        let _b: bool = xu32 == yu32;
        let _b: bool = xu32 > yu32;
        let _b: bool = xu32 >= yu32;
        let _b: bool = xu32 < yu32;
        let _b: bool = xu32 <= yu32;

        let xi32: i32 = 1;
        let yi32: i32 = 2;
        let _zi32 = xi32 / yi32;

        let _b: bool = xi32 == yi32;
        let _b: bool = xi32 > yi32;
        let _b: bool = xi32 >= yi32;
        let _b: bool = xi32 < yi32;
        let _b: bool = xi32 <= yi32;

        let an_f32: f32 = 1.0;
        let another_f32: f32 = 2.0;
        let _yet_another_f32: f32 = an_f32 / another_f32;

        let _b: bool = an_f32 == another_f32;
        let _b: bool = an_f32 > another_f32;
        let _b: bool = an_f32 >= another_f32;
        let _b: bool = an_f32 < another_f32;
        let _b: bool = an_f32 <= another_f32;

        // Convert things.
        let x: f32 = 0.0;
        let x: i32 = convert_scalar::<i32>(x);
        let x: f32 = convert_scalar::<f32>(x);
        let x: f64 = convert_scalar::<f64>(x);
        let _x: f16 = convert_scalar::<f16>(x);

        let shape: Shape<{ [128, 256] }> = Shape::<{ [128, 256] }> {
            dims: &[num_pid.0, 256i32],
        };
        let shape_dim_1: i32 = 128;
        let shape_dim_2: i32 = 256;
        let stride_dim_1: i32 = num_pid.0;
        let stride: Array<{ [-1, 128] }> = Array::<{ [-1, 128] }> {
            dims: &[stride_dim_1],
        };
        let dynamic_shape: Shape<{ [-1, -1] }> = Shape::<{ [-1, -1] }> {
            dims: &[shape_dim_1, shape_dim_2],
        };

        unsafe {
            // Basic loop pattern with a tile.
            let token: Token = new_token_unordered();
            let _some_tensor: Tensor<f32, { [-1, -1] }> =
                make_tensor_view(ptr_tile, dynamic_shape, stride, token);
            let mut partition: PartitionMut<f32, { [128, 256] }> =
                make_partition_view_mut(&y, shape, token);
            let idx: [i32; 2] = [0i32, 0i32];
            let mut some_tile: Tile<f32, { [128, 256] }> = load_from_view_mut(&partition, idx);
            store_to_view_mut(&mut partition, some_tile, idx);
            store_to_view_mut(&mut partition, some_tile, idx);
            for _i in 0i32..10i32 {
                let some_tile_2: Tile<f32, { [128, 256] }> = constant(2.0, shape);
                some_tile = some_tile + some_tile_2;
                store_to_view_mut(&mut partition, some_tile, idx);
                continue;
            }
            let some_3: Tile<f32, { [128, 256] }> = some_tile + some_tile;
            store_to_view_mut(&mut partition, some_3, idx);
        }

        let _basic_string = "a string.";

        let x: [i32; 2] = [1i32, 2i32];
        let _x_val: i32 = x[0];
        let x: &[i32] = &[1i32, 2i32];
        let _x_val: i32 = x[0];

        const ARRAY: [i32; 2] = [1, 2];
        const X0: i32 = ARRAY[0];
        let _x1: i32 = ARRAY[1];

        if shape_dim_1 != shape_dim_2 {
            cuda_tile_assert!(shape_dim_1 != shape_dim_2, "Impossible");
        } else {
            cuda_tile_assert!(shape_dim_1 == shape_dim_2, "Impossible");
        }
    }

    #[cutile::entry()]
    fn negative_constant_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let shape = output.shape();
        let _neg_float: Tile<f32, S> = constant(-1.0, shape);
        let _neg_int: Tile<i32, S> = constant(-42i32, shape);
        let _neg_suffixed: Tile<f32, S> = constant(-2.5f32, shape);
    }
}

use basics_and_inlining_module::_module_asts;

#[test]
fn compile_inlining() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "basics_and_inlining_module",
            "inlining_kernel",
            &[
                "f32".to_string(),
                1.to_string(),
                128.to_string(),
                256.to_string(),
                512.to_string(),
                2.to_string(),
            ],
            &[("y", &[1024, 1, 1])],
            None,
            gpu_name,
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("{module_op_str}");
    });
}

#[test]
fn compile_basics() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "basics_and_inlining_module",
            "basics_kernel",
            &[128.to_string(), 256.to_string(), 512.to_string()],
            &[("y", &[1024, 1]), ("w", &[1, 2, 3])],
            None,
            gpu_name,
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("{module_op_str}");
    });
}

#[test]
fn compile_negative_constant() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "basics_and_inlining_module",
            "negative_constant_kernel",
            &[128.to_string()],
            &[("output", &[1024])],
            None,
            gpu_name,
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed to compile negative constant kernel.")
            .as_operation()
            .to_string();
        assert!(module_op_str.contains("-1.0"));
        println!("{module_op_str}");
    });
}
