/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA kernel launch builder with argument marshalling.

use crate::device_context::with_default_device_policy;
use crate::device_future::DeviceFuture;
use crate::device_operation::{DeviceOperation, ExecutionContext};
use crate::error::DeviceError;
use crate::scheduling_policies::SchedulingPolicy;
use anyhow::{Context, Result};
use cuda_core::{launch_kernel, CudaFunction, CudaStream, LaunchConfig};
use std::ffi::c_void;
use std::fmt::Debug;
use std::future::IntoFuture;
use std::sync::Arc;
use std::vec::Vec;

/// A builder for asynchronously launching a CUDA kernel on a stream.
#[derive(Debug)]
pub struct AsyncKernelLaunch {
    pub func: Arc<CudaFunction>,
    pub args: Vec<*mut c_void>,
    cfg: Option<LaunchConfig>,
}

unsafe impl Send for AsyncKernelLaunch {}

impl Drop for AsyncKernelLaunch {
    fn drop(&mut self) {
        let _ = self
            .args
            .iter()
            .map(|&arg| {
                // Reconstruct the boxes. Pointers will be dropped when they go out of scope.
                unsafe { Box::from_raw(arg as *mut usize) }
            })
            .collect::<Vec<_>>();
    }
}

impl AsyncKernelLaunch {
    /// Creates a new kernel launch builder for the given CUDA function.
    pub fn new(func: Arc<CudaFunction>) -> AsyncKernelLaunch {
        AsyncKernelLaunch {
            func,
            args: Vec::new(),
            cfg: None,
        }
    }

    /// Pushes a kernel argument by value.
    #[inline(always)]
    pub fn push_arg<T: KernelArgument>(&mut self, arg: T) -> &mut Self {
        arg.push_arg(self);
        self
    }

    /// Pushes a kernel argument from an `Arc` reference.
    #[inline(always)]
    pub fn push_arg_arc<T: ArcKernelArgument>(&mut self, arg: &Arc<T>) -> &mut Self {
        arg.push_arg_arc(self);
        self
    }

    /// Sets the grid/block dimensions and shared memory configuration for the launch.
    pub fn set_launch_config(&mut self, cfg: LaunchConfig) -> &mut Self {
        self.cfg = Some(cfg);
        self
    }

    /// Launches the kernel on the given CUDA stream.
    ///
    /// # Safety
    /// The caller must ensure the kernel arguments and launch config are valid.
    unsafe fn launch(mut self, stream: &Arc<CudaStream>) -> Result<(), DeviceError> {
        let cfg = self.cfg.ok_or(DeviceError::Launch(
            "Await called before launching the kernel.".to_string(),
        ))?;
        launch_kernel(
            self.func.cu_function(),
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            stream.cu_stream(),
            &mut self.args,
        )
        .with_context(|| {
            format!(
                r#"
                Failed to launch kernel.
                args: {:#?}
                cfg: {:#?}"#,
                self.args, cfg
            )
        })?;
        Ok(())
    }
}

/// A kernel argument that can be pushed from an `Arc` reference.
pub trait ArcKernelArgument {
    // #[inline(always)] Dont think this is necessary. This will be deprecated for required trait methods
    fn push_arg_arc(self: &Arc<Self>, launcher: &mut AsyncKernelLaunch);
}

/// A kernel argument that can be pushed by value into an `AsyncKernelLaunch`.
pub trait KernelArgument {
    // #[inline(always)] Dont think this is necessary. This will be deprecated for required trait methods
    fn push_arg(self, launcher: &mut AsyncKernelLaunch);
}

impl<T> KernelArgument for Box<T> {
    fn push_arg(self, launcher: &mut AsyncKernelLaunch) {
        let r = Box::into_raw(self);
        launcher.args.push(r as *mut _);
    }
}

impl DeviceOperation for AsyncKernelLaunch {
    type Output = ();

    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        self.launch(ctx.get_cuda_stream())
    }
}

impl IntoFuture for AsyncKernelLaunch {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), AsyncKernelLaunch>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}
