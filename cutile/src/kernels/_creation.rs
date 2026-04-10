/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/// Tensor creation and initialization kernels.
///
/// This module contains GPU kernels for creating and initializing tensors with
/// specific patterns. These kernels are used internally by the `api` module functions
/// like `zeros`, `ones`, `full`, and `arange`.

#[crate::module(tile_rust_crate = true)]
pub mod creation {
    use crate::core::*;

    /// Fills a tensor with a constant value.
    ///
    /// This kernel broadcasts a scalar value to fill an entire tensor partition.
    /// Each thread block processes one partition.
    ///
    /// ## Parameters
    ///
    /// - `value`: The constant value to fill the tensor with
    /// - `tensor`: Mutable tensor to store the result
    ///
    /// ## Type Parameters
    ///
    /// - `T`: Element type (must implement `ElementType`)
    /// - `S`: Partition shape (compile-time constant)
    ///
    /// ## Usage
    ///
    /// This kernel is used by `api::full()`, `api::zeros()`, and `api::ones()`.
    ///
    /// ```rust,ignore
    /// use cutile::kernels::creation::full;
    ///
    /// let val = 42.0f32;
    /// let tensor = api::zeros(&[1024]).partition([128]);
    /// let result = value((val, tensor))
    ///     .then(full)
    ///     .unzip();
    /// ```
    #[crate::entry()]
    pub fn full<T: ElementType, const S: [i32; 1]>(value: T, tensor: &mut Tensor<T, S>) {
        let value_tile: Tile<T, S> = value.broadcast(tensor.shape());
        tensor.store(value_tile);
    }

    /// Creates a sequence of consecutive integers starting from the partition offset.
    ///
    /// This kernel generates values like [0, 1, 2, ...] across all partitions,
    /// with each partition computing its portion based on the thread block ID.
    ///
    /// ## Parameters
    ///
    /// - `tensor`: Mutable tensor to store the sequence
    ///
    /// ## Type Parameters
    ///
    /// - `T`: Element type (must implement `ElementType`)
    /// - `S`: Partition shape (compile-time constant)
    ///
    /// ## Usage
    ///
    /// This kernel is used by `api::arange()`.
    ///
    /// ```rust,ignore
    /// use cutile::kernels::creation::arange;
    ///
    /// let tensor = Tensor::<i32>::uninitialized(256)
    ///     .await
    ///     .partition([64]);
    /// let result = value((tensor,))
    ///     .then(arange)
    ///     .unzip();
    /// // Result contains [0, 1, 2, ..., 255]
    /// ```
    #[crate::entry()]
    pub fn arange<T: ElementType, const S: [i32; 1]>(tensor: &mut Tensor<T, S>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let pdim: (i32, i32, i32) = get_tile_block_id();
        let offset = pid.0 * pdim.0;
        let offset_tile = broadcast_scalar(offset, tensor.shape());
        let range = iota::<i32, S>(tensor.shape());
        let offset_range = offset_tile + range;
        let offset_range: Tile<T, S> = convert_tile(offset_range);
        tensor.store(offset_range);
    }
}
