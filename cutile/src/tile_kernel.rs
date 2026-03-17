/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tile kernel compilation, caching, launching, and partitioning for CUDA device operations.

use anyhow::{Context, Result};
use candle_core::WithDType;
use cuda_async::error::DeviceError;
use cuda_core::{memcpy_dtoh_async, CudaFunction};
use cutile_compiler::ast::Module;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile::ModuleOperation;
use cutile_compiler::cuda_tile_runtime_utils::{compile_module, get_gpu_name};
use std::alloc::{alloc, Layout};
use std::fs;
use std::future::IntoFuture;
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::*;
use crate::tensor::{IntoPartition, IntoPartitionArc, Partition, Tensor};

pub use cuda_async::{
    device_box::*, device_context::*, device_future::*, device_operation::*, launch::*,
    scheduling_policies::*,
};

/// Cache key for a compiled tile kernel.
///
/// Two kernel invocations that share the same `TileFunctionKey` can reuse the same compiled
/// CUDA module and function, avoiding recompilation. The key captures everything that can
/// change the generated GPU code: module name, function name, generic type/const parameters,
/// tensor stride layouts, and (optionally) the launch grid.
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct TileFunctionKey {
    module_name: String,
    function_name: String,
    pub function_generics: Vec<String>,
    pub stride_args: Vec<(String, Vec<i32>)>,
    pub grid: Option<(u32, u32, u32)>,
}

impl TileFunctionKey {
    pub fn new(
        module_name: String,
        function_name: String,
        function_generics: Vec<String>,
        stride_args: Vec<(String, Vec<i32>)>,
        grid: Option<(u32, u32, u32)>,
    ) -> Self {
        Self {
            module_name,
            function_name,
            function_generics,
            stride_args,
            grid,
        }
    }
}

impl FunctionKey for TileFunctionKey {}

/// Reads IR (MLIR or PTX) from a file.
///
/// This helper function reads intermediate representation files from disk, typically
/// for debugging purposes when using `use_debug_mlir` or similar options.
///
/// ## Parameters
///
/// - `path`: Path to the IR file to read
///
/// ## Returns
///
/// The file contents as a UTF-8 string, or an I/O error if reading fails.
#[expect(unused)]
fn read_ir(path: String) -> Result<String, std::io::Error> {
    let s = String::from_utf8(fs::read(path)?).expect("Unable to convert from utf8 to string.");
    Ok(s)
}

/// Writes IR (MLIR or PTX) to a file for debugging.
///
/// This helper function writes intermediate representation to disk when kernel functions
/// are marked with `dump_mlir_dir` or `dump_ptx_dir` entry attributes. The filename
/// includes the module name, function name, and cache hash for uniqueness.
///
/// ## Parameters
///
/// - `module_name`: Name of the module containing the kernel
/// - `function_name`: Name of the kernel function
/// - `cache_hash_str`: Unique hash identifying this compilation
/// - `extension`: File extension (e.g., "mlir", "ptx")
/// - `dir`: Directory to write the file to
/// - `contents`: IR contents to write
///
/// ## Panics
///
/// Panics if the file cannot be written.
fn write_ir(
    module_name: &str,
    function_name: &str,
    cache_hash_str: &str,
    extension: &str,
    dir: &str,
    contents: &str,
) {
    let filename = format!("{module_name}_{function_name}_{cache_hash_str}.{extension}");
    let path = PathBuf::from(dir).join(filename);
    fs::write(path.clone(), contents).unwrap_or_else(|_| panic!("Failed to write {path:?}")); // Writes the string as bytes
    println!("IR written to {path:?}");
}

/// Compiles a tile function to CUDA and caches it for reuse.
///
/// Handles the complete compilation pipeline from Rust/MLIR to CUDA:
/// 1. Checks the thread-local cache for a previously compiled function
/// 2. If not cached, compiles the module AST to MLIR, then to PTX/CUBIN
/// 3. Loads the compiled function and caches it for future use
///
/// The caching key is based on the module name, function name, type generics, stride arguments,
/// and compile-time grid dimensions, ensuring correct reuse across different specializations.
///
/// ## Arguments
///
/// * `ctx` - Execution context containing device information
/// * `module_asts` - Closure that produces the AST modules to compile
/// * `module_name` - Name of the module containing the function
/// * `function_name` - Name of the function to compile
/// * `function_entry` - Entry point name in the compiled CUDA code
/// * `function_generics` - Type and const generic arguments (e.g., `["f32", "256"]`)
/// * `stride_args` - Stride information for tensor arguments
/// * `const_grid` - Optional compile-time constant grid dimensions
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::tile_async::compile_from_context;
///
/// let ctx = get_execution_context();
/// let function = compile_from_context(
///     &ctx,
///     || vec![my_module_ast()],
///     "my_module",
///     "my_function",
///     "my_function_kernel",
///     vec!["f32".to_string(), "128".to_string()],
///     vec![],
///     None
/// );
/// ```
#[allow(clippy::too_many_arguments)]
pub fn compile_from_context<F: Fn() -> Vec<Module>>(
    ctx: &ExecutionContext,
    module_asts: F,
    module_name: &str,
    function_name: &str,
    function_entry: &str,
    function_generics: Vec<String>,
    stride_args: Vec<(String, Vec<i32>)>,
    const_grid: Option<(u32, u32, u32)>,
) -> Result<(Arc<CudaFunction>, Arc<Validator>), Error> {
    let device_id: usize = ctx.get_device_id();
    // Compilation constructs a lookup key.
    let key = TileFunctionKey::new(
        module_name.to_string(),
        function_name.to_string(),
        function_generics,
        stride_args,
        const_grid,
    );
    let cache_hash_str = key.get_hash_string();
    if contains_cuda_function(device_id, &key) {
        // A hit to the thread local kernel cache returns the compiled function.
        let func = get_cuda_function(device_id, &key)?;
        let validator = get_function_validator(device_id, &key)?;
        Ok((func, validator))
    } else {
        let gpu_name = get_gpu_name(device_id);
        // A miss compiles, caches, and returns the compiled function.
        let modules = CUDATileModules::new(module_asts())?;
        let debug_mlir_path = modules.get_entry_arg_string_by_function_name(
            module_name,
            function_name,
            "use_debug_mlir",
        )?;
        // TODO (hme): Re-enable some debug support for internal.
        // let mlir = if let Some(debug_mlir_path) = &debug_mlir_path {
        //     println!("USING DEBUG MLIR: {debug_mlir_path}");
        //     let mlir = read_ir(debug_mlir_path.to_string()).expect("Failed to read debug MLIR.");
        //     mlir
        // } else {
        //     let module_op: ModuleOperation = compiler.compile(
        //         module_name,
        //         function_name,
        //         &key.function_generics,
        //         &key.stride_args
        //             .iter()
        //             .map(|x| (x.0.as_str(), x.1.as_slice()))
        //             .collect::<Vec<_>>(),
        //         const_grid,
        //         gpu_name.clone(),
        //     );
        //     let mlir = module_op.as_operation().to_string();
        //     mlir
        // };
        let args = (
            module_name,
            function_name,
            &key.function_generics,
            &key.stride_args
                .iter()
                .map(|x| (x.0.as_str(), x.1.as_slice()))
                .collect::<Vec<_>>(),
            const_grid,
            gpu_name.clone(),
        );
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            args.0,
            args.1,
            args.2,
            args.3,
            args.4,
            args.5.clone(),
        )?;
        let validator: Validator = compiler.get_validator();
        let validator = Arc::new(validator);
        let module_op: ModuleOperation = compiler.compile()?;
        let mlir = module_op.as_operation().to_string();
        if modules.get_entry_arg_bool_by_function_name(module_name, function_name, "print_ir")? {
            if debug_mlir_path.is_some() {
                println!("LOADED MLIR: {module_name}::{function_name}\n{}", mlir);
            } else {
                println!("COMPILED MLIR: {module_name}::{function_name}\n{}", mlir);
            }
        }
        if let Some(path) = modules.get_entry_arg_string_by_function_name(
            module_name,
            function_name,
            "dump_mlir_dir",
        )? {
            write_ir(
                module_name,
                function_name,
                cache_hash_str.as_str(),
                "mlir",
                path.as_str(),
                mlir.as_str(),
            );
        }
        let cubin_filename = compile_module(&module_op, &gpu_name);
        // if let Some(path) = compiler.get_entry_arg_string_by_function_name(
        //     module_name,
        //     function_name,
        //     "dump_ptx_dir",
        // ) {
        //     write_ir(
        //         module_name,
        //         function_name,
        //         cache_hash_str.as_str(),
        //         "ptx",
        //         path.as_str(),
        //         ptx.as_str(),
        //     );
        // }
        // if compiler.get_entry_arg_bool_by_function_name(module_name, function_name,"print_ir") {
        //     println!("COMPILED PTX: {module_name}::{function_name}");
        //     println!("{ptx}");
        //     println!();
        // }
        let module = load_module_from_file(&cubin_filename, device_id)?;
        let function = Arc::new(
            module
                .load_function(function_entry)
                .expect("Failed to compile function."),
        );
        insert_cuda_function(device_id, &key, (module, function.clone()))?;
        insert_function_validator(device_id, &key, validator.clone())?;
        Ok((function, validator))
    }
}

/// Validates that all partition grids match the expected launch grid.
pub fn validate_grids(
    grid: (u32, u32, u32),
    partition_grids: &[(u32, u32, u32)],
) -> Result<(), Error> {
    // Make sure we're not trying to map mutable references to incorrect launch grid.
    if let Some(partition_grid) = partition_grids.iter().find(|&&i| i != grid) {
        Err(Error::KernelLaunch(KernelLaunchError(format!(
            "{:?} != {:?}",
            grid, partition_grid
        ))))
    } else {
        Ok(())
    }
}

/// Infers the launch grid for a kernel from partitioned tensor inputs.
///
/// If a grid is explicitly specified (non-zero), it is used directly. Otherwise, the grid
/// is inferred from partitioned tensor inputs. All inferred grids must match, or the
/// function will panic.
///
/// ## Panics
///
/// Panics if no grid is specified and no inferred grids are available, or if inferred
/// grids from different inputs don't match.
pub fn infer_launch_grid(
    grid: (u32, u32, u32),
    inferred_grids: &[(u32, u32, u32)],
) -> Result<(u32, u32, u32), Error> {
    if grid != (0, 0, 0) {
        // A launch grid was specified.
        if !inferred_grids.is_empty() {
            validate_grids(grid, inferred_grids).with_context(|| {
                "Specified launch grid does not match inferred tensor partition grid"
            })?;
        }
        return Ok(grid);
    }
    // Try to infer launch grid.
    if inferred_grids.is_empty() {
        return kernel_launch_error_result("Launch grid required.");
    }
    let grid = inferred_grids[0];
    validate_grids(grid, inferred_grids)
        .with_context(|| "Inferred tensor partition grids do not match")?;
    Ok(grid)
}

/// A compiled CUDA kernel generated from Rust code that can be launched on the GPU.
///
/// `TileKernel` extends `DeviceOperation` with kernel-specific functionality. Kernels are
/// automatically generated from Rust functions marked with `#[cutile::entry]` and compiled
/// to MLIR, then to CUDA PTX at runtime.
///
/// The trait provides methods for configuring kernel launch parameters such as grid dimensions,
/// type generics, and shared memory. Grid dimensions can be set explicitly or inferred from
/// partitioned tensor inputs.
///
/// ## Examples
///
/// ### Basic kernel launch
///
/// ```rust,ignore
/// use cutile::tile_async::TileKernel;
/// use cuda_async::device_operation::DeviceOperation;
///
/// #[cutile::module]
/// mod my_module {
///     use cutile::core::*;
///
///     #[cutile::entry]
///     fn hello_world() {
///         let pid = get_tile_block_id();
///         cuda_tile_print!("Hello from block {}\n", pid.0);
///     }
/// }
///
/// // Launch with explicit grid
/// my_module::hello_world_sync()
///     .grid((4, 1, 1))
///     .sync_on(&stream);
/// ```
///
/// ### Kernel with arguments and grid inference
///
/// ```rust,ignore
/// #[cutile::module]
/// mod kernels {
///     use cutile::core::*;
///
///     #[cutile::entry]
///     fn add<T: ElementType, const N: i32>(
///         z: &mut Tensor<T, {[N]}>,
///         x: &Tensor<T, {[N]}>,
///         y: &Tensor<T, {[N]}>,
///     ) {
///         let tile_x = load_tile_mut(x);
///         let tile_y = load_tile_mut(y);
///         z.store(tile_x + tile_y);
///     }
/// }
///
/// use kernels::add_apply;
///
/// // Grid is inferred from partitioned tensors
/// let x = api::ones([256]).partition([64]);
/// let y = api::ones([256]).partition([64]);
/// let z = api::zeros([256]).partition([64]);
///
/// let result = zip!(z, x, y)
///     .apply(add_apply)
///     .generics(vec!["f32".to_string(), "256".to_string()])
///     .await; // Automatically launches with grid (4, 1, 1)
/// ```
///
/// ### Using with async composition
///
/// ```rust,ignore
/// async fn pipeline() -> impl DeviceOperation<Output=Tensor<f32>> {
///     let x = api::randn(0.0, 1.0, [128, 128]).await;
///
///     // Chain kernel operations
///     let y = my_kernel_1(x.clone())
///         .grid((8, 8, 1))
///         .await;
///
///     let z = my_kernel_2(y)
///         .grid((4, 4, 1))
///         .await;
///
///     z
/// }
/// ```
pub trait TileKernel<ARGS: Send, DI>: DeviceOperation<Output = ARGS>
where
    DI: DeviceOperation<Output = ARGS>,
{
    /// Compiles the kernel from module ASTs, returning the CUDA function and validator.
    #[allow(clippy::too_many_arguments)]
    fn compile<F: Fn() -> Vec<Module>>(
        &mut self,
        ctx: &ExecutionContext,
        module_asts: F,
        module_name: &str,
        function_name: &str,
        function_entry: &str,
        function_generics: Vec<String>,
        stride_args: Vec<(String, Vec<i32>)>,
        grid: Option<(u32, u32, u32)>,
    ) -> Result<(Arc<CudaFunction>, Arc<Validator>), Error> {
        compile_from_context(
            ctx,
            module_asts,
            module_name,
            function_name,
            function_entry,
            function_generics,
            stride_args,
            grid,
        )
    }
    /// Sets the type and const generic arguments for this kernel.
    fn generics(self, generics: Vec<String>) -> Self;
    /// Sets a compile-time constant grid, enabling grid-dependent optimizations.
    fn const_grid(self, grid: (u32, u32, u32)) -> Self;
    /// Sets the runtime launch grid dimensions.
    fn grid(self, grid: (u32, u32, u32)) -> Self;
    /// Infers the launch grid from partitioned tensor inputs, or uses the explicit grid.
    fn infer_launch_grid(
        &self,
        inferred_grids: &[(u32, u32, u32)],
    ) -> Result<(u32, u32, u32), Error> {
        let grid = self.get_launch_grid();
        infer_launch_grid(grid, inferred_grids)
    }
    /// Returns the currently configured launch grid dimensions.
    fn get_launch_grid(&self) -> (u32, u32, u32);
    /// Returns the dynamic shared memory size in bytes. Defaults to 0.
    fn get_launch_smem(&self) -> u32 {
        0
    }
    /// Returns the thread block dimensions. Defaults to `(1, 1, 1)`.
    fn get_launch_block(&self) -> (u32, u32, u32) {
        (1, 1, 1)
    }
    // fn validate(validator: &Validator) -> Result<(), Error> {

    // }
    // fn validate_arc<T: WithDType>(
    //     &self,
    //     func_name: String,
    //     var_name: String,
    //     arc: &Arc<Tensor<T>>,
    //     shape: &[i32],
    // ) -> Result<(), KernelLauncherError> {
    //     let input_shape = &arc.shape;
    //     if input_shape != shape {
    //         return Err(KernelLauncherError::InvalidTensorShape(format!(
    //             "Unexpected shape {:?} for argument {} for function {}.",
    //             input_shape, var_name, func_name
    //         )));
    //     }
    //     Ok(())

    //     // if input_shape.len() != shape.len() {
    //     //     return Err(KernelLauncherError::InvalidTensorShape(format!("Unexpected rank {} for argument {} for function {}.",
    //     //         input_shape.len(),
    //     //         var_name,
    //     //         func_name
    //     //     )));
    //     // }
    //     // for i in 0..input_shape.len() {
    //     //     let input_dim = input_shape[i];
    //     //     let param_dim = shape[i];
    //     //     if param_dim == -1 {
    //     //         continue;
    //     //     }
    //     //     if input_dim != param_dim {
    //     //         return Err(KernelLauncherError::InvalidTensorShape(format!("Unexpected rank {} for argument {} for function {}.",
    //     //             input_shape.len(),
    //     //             var_name,
    //     //             func_name
    //     //         )));
    //     //     }
    //     // }
    // }
}

/// Implements kernel argument passing for `Tensor` when wrapped in `Arc`.
///
/// Pushes the device pointer, shape, and stride information to the kernel launcher
/// in the order expected by compiled tile functions.
impl<T: WithDType> ArcKernelArgument for Tensor<T> {
    fn push_arg_arc(self: &Arc<Self>, launcher: &mut AsyncKernelLaunch) {
        launcher.push_arg(Box::new(self.cu_deviceptr()));
        for dim in self.shape.iter() {
            launcher.push_arg(Box::new(*dim));
        }
        for stride in self.strides.iter() {
            launcher.push_arg(Box::new(*stride));
        }
    }
}

/// Implements kernel argument passing for partitioned tensors.
///
/// Pushes the device pointer, tensor shape and strides, followed by partition shape
/// and strides. This allows kernels to access both the full tensor and the partition
/// information for block-level indexing.
impl<T: WithDType> KernelArgument for &Partition<Tensor<T>> {
    fn push_arg(self, launcher: &mut AsyncKernelLaunch) {
        launcher.push_arg(Box::new(self.object.cu_deviceptr()));
        for dim in self.object.shape.iter() {
            launcher.push_arg(Box::new(*dim));
        }
        for stride in self.object.strides.iter() {
            launcher.push_arg(Box::new(*stride));
        }
        for dim in self.partition_shape.iter() {
            launcher.push_arg(Box::new(*dim));
        }
        for stride in self.partition_strides.iter() {
            launcher.push_arg(Box::new(*stride));
        }
    }
}

// Partition

/// Extension trait that enables partitioning device operations into tiles.
///
/// This trait allows async operations that produce tensors to be partitioned before
/// execution, enabling automatic grid inference for tile kernels. The partition divides
/// the tensor into blocks that map to CUDA thread blocks.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::tile_async::IntoDeviceOperationPartition;
///
/// // Partition a tensor operation before it executes
/// let x = api::ones([1024]).partition([128]);  // Creates 8 partitions
///
/// // Use partitioned tensors with kernels for automatic grid inference
/// let y = api::randn(0.0, 1.0, [256, 256]).partition([64, 64]);  // 4x4 grid
/// let result = my_kernel(y).await;  // Grid (4, 4, 1) inferred automatically
/// ```
pub trait IntoDeviceOperationPartition<I, DI>
where
    I: Send + IntoPartition + IntoPartitionArc,
    DI: DeviceOperation<Output = I>,
{
    /// Partitions the output of this device operation into tiles of the given shape.
    ///
    /// The partition shape determines how the tensor is divided across CUDA thread blocks.
    fn partition<const RANK: usize>(
        self,
        partition_shape: [i32; RANK],
    ) -> DeviceOperationPartition<RANK, I, DI>;
}

impl<I, DI> IntoDeviceOperationPartition<I, DI> for DI
where
    I: Send + IntoPartition + IntoPartitionArc,
    DI: DeviceOperation<Output = I>,
{
    fn partition<const RANK: usize>(
        self,
        partition_shape: [i32; RANK],
    ) -> DeviceOperationPartition<RANK, I, DI>
    where
        Self: Sized,
    {
        DeviceOperationPartition::<RANK, I, DI> {
            partition_shape,
            op: self,
        }
    }
}

/// A device operation that partitions its output into tiles.
///
/// This wrapper executes the underlying device operation and then partitions its result
/// according to the specified partition shape. The resulting partitioned tensor can be
/// used with tile kernels to automatically infer launch grid dimensions.
///
/// Created by calling `.partition()` on any device operation that produces a partitionable output.
///
/// ## Examples
///
/// ```rust,ignore
/// // Create a partitioned tensor asynchronously
/// let x = api::zeros([1024]).partition([64]);  // DeviceOperationPartition
/// let partitioned = x.await;  // Partition<Tensor<f32>>
///
/// // Use with kernel that accepts partitioned inputs
/// let result = zip!(partitioned, other_partitioned)
///     .apply(my_kernel_apply)
///     .await;
/// ```
pub struct DeviceOperationPartition<const RANK: usize, I, DI>
where
    I: Send + IntoPartition + IntoPartitionArc,
    DI: DeviceOperation<Output = I>,
{
    partition_shape: [i32; RANK],
    op: DI,
}

unsafe impl<const RANK: usize, I, DI> Send for DeviceOperationPartition<RANK, I, DI>
where
    I: Send + IntoPartition + IntoPartitionArc,
    DI: DeviceOperation<Output = I>,
{
}

impl<const RANK: usize, I, DI> DeviceOperation for DeviceOperationPartition<RANK, I, DI>
where
    I: Send + IntoPartition + IntoPartitionArc,
    DI: DeviceOperation<Output = I>,
{
    type Output = Partition<I>;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let val = self.op.execute(context)?;
        Ok(val.partition(self.partition_shape))
    }
}

impl<const RANK: usize, I, DI> IntoFuture for DeviceOperationPartition<RANK, I, DI>
where
    I: Send + IntoPartition + IntoPartitionArc,
    DI: DeviceOperation<Output = I>,
{
    type Output = Result<Partition<I>, DeviceError>;
    type IntoFuture = DeviceFuture<Partition<I>, DeviceOperationPartition<RANK, I, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

// Unwrap Partition

/// A device operation that unwraps a partitioned tensor back to a regular tensor.
///
/// This operation removes the partition structure from a tensor, converting a
/// `Partition<Tensor<T>>` back to `Tensor<T>`. This is useful after kernel operations
/// that work on partitioned inputs but need to return regular tensors for further
/// processing.
///
/// Created by calling `unwrap_partition()` on a device operation that produces a partition.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::tile_async::unwrap_partition;
///
/// // After a kernel operation on partitioned tensors
/// let x = api::ones([256]).partition([64]);
/// let y = my_kernel(x).await;  // Returns Partition<Tensor<f32>>
///
/// // Unwrap back to a regular tensor
/// let z = unwrap_partition(y).await;  // Now Tensor<f32>
/// ```
pub struct UnwrapPartition<I: Send, DI>
where
    DI: DeviceOperation<Output = Partition<I>>,
{
    pub(crate) op: DI,
}

unsafe impl<I: Send, DI> Send for UnwrapPartition<I, DI> where
    DI: DeviceOperation<Output = Partition<I>>
{
}

impl<I: Send, DI> DeviceOperation for UnwrapPartition<I, DI>
where
    DI: DeviceOperation<Output = Partition<I>>,
{
    type Output = I;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let val = self.op.execute(context)?;
        Ok(val.unpartition())
    }
}

impl<I: Send, DI> IntoFuture for UnwrapPartition<I, DI>
where
    DI: DeviceOperation<Output = Partition<I>>,
{
    type Output = Result<I, DeviceError>;
    type IntoFuture = DeviceFuture<I, UnwrapPartition<I, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Unwraps a partitioned device operation back to a regular tensor operation.
///
/// Converts a device operation that produces a `Partition<T>` into one
/// that produces `T` directly. Useful for converting partitioned kernel outputs
/// back to regular tensors for further processing.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::tile_async::unwrap_partition;
///
/// async fn process_data() -> Tensor<f32> {
///     let x = api::randn(0.0, 1.0, [1024]).partition([128]);
///     let processed = my_tiled_kernel(x);  // Returns Partition<Tensor<f32>>
///
///     // Unwrap to get a regular tensor
///     unwrap_partition(processed).await
/// }
/// ```
pub fn unwrap_partition<I: Send, DI>(op: DI) -> UnwrapPartition<I, DI>
where
    DI: DeviceOperation<Output = Partition<I>>,
{
    UnwrapPartition { op }
}

// ToHostVec

/// A device operation that copies a tensor from device memory to a host `Vec<T>`.
pub struct TensorToHostVec<T: WithDType, DI>
where
    DI: DeviceOperation<Output = Tensor<T>>,
{
    pub(crate) op: DI,
}

unsafe impl<T: WithDType, DI> Send for TensorToHostVec<T, DI> where
    DI: DeviceOperation<Output = Tensor<T>>
{
}

impl<T: WithDType, DI> DeviceOperation for TensorToHostVec<T, DI>
where
    DI: DeviceOperation<Output = Tensor<T>>,
{
    type Output = Vec<T>;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let tensor = self.op.execute(context)?;
        let cu_deviceptr = tensor.device_box.cu_deviceptr();
        let size = tensor.size();
        let layout = Layout::array::<T>(size).expect("overflow cannot happen");
        let async_ptr = unsafe { alloc(layout).cast::<T>() };
        memcpy_dtoh_async(async_ptr, cu_deviceptr, size, context.get_cuda_stream());
        Ok(unsafe { Vec::from_raw_parts(async_ptr, size, size) })
    }
}

impl<T: WithDType, DI> IntoFuture for TensorToHostVec<T, DI>
where
    DI: DeviceOperation<Output = Tensor<T>>,
{
    type Output = Result<Vec<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Vec<T>, TensorToHostVec<T, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Extension trait for converting a tensor device operation into a host `Vec<T>` operation.
pub trait TensorDeviceOpToHostVec<T: WithDType> {
    /// Wraps this operation to copy the resulting tensor to a host `Vec<T>`.
    fn to_host_vec(self) -> impl DeviceOperation<Output = Vec<T>>
    where
        Self: DeviceOperation<Output = Tensor<T>>,
    {
        TensorToHostVec { op: self }
    }
}

impl<T: WithDType, DI> TensorDeviceOpToHostVec<T> for DI where
    DI: DeviceOperation<Output = Tensor<T>>
{
}
