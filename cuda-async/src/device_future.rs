/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Future type that bridges CUDA stream callbacks with Rust's async executor.

use crate::device_operation::{DeviceOp, ExecutionContext};
use crate::error::DeviceError;
use futures::task::AtomicWaker;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

/// State machine for tracking the lifecycle of a device future.
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum DeviceFutureState {
    // The future was created with an error and will resolve immediately on first poll.
    /// The future was created with an error and will resolve immediately.
    Failed,
    // The stream operation has not yet been scheduled. No callback has been added.
    /// The stream operation has not yet been scheduled.
    Idle,
    // The stream operation has been scheduled and a callback has been added to the stream.
    // The callback should be added such that it immediately succeeds the scheduled operation.
    /// The stream operation is in-flight and a completion callback is registered.
    Executing,
    // The callback has been fired, indicating the completion of the stream operation.
    /// The stream callback has fired, indicating the operation is done.
    Complete,
}

/// Shared state between a CUDA stream callback and the async waker.
#[derive(Debug)]
pub struct StreamCallbackState {
    pub(crate) waker: AtomicWaker,
    pub(crate) complete: AtomicBool,
}

impl StreamCallbackState {
    /// Creates a new callback state with the completion flag unset.
    pub fn new() -> Self {
        Self {
            waker: AtomicWaker::new(),
            complete: AtomicBool::new(false),
        }
    }
    /// Marks the operation as complete and wakes the associated task.
    pub fn signal(&self) {
        self.complete.store(true, Ordering::Relaxed);
        self.waker.wake();
    }
}

/// A future that executes a [`DeviceOp`] on a CUDA stream and resolves upon completion.
#[derive(Debug)]
pub struct DeviceFuture<T: Send, DO: DeviceOp<Output = T>> {
    pub(crate) device_operation: Option<DO>,
    pub(crate) execution_context: Option<ExecutionContext>,
    pub(crate) result: Option<T>,
    pub(crate) error: Option<DeviceError>,
    pub(crate) state: DeviceFutureState,
    pub(crate) callback_state: Option<Arc<StreamCallbackState>>,
}

impl<T: Send, DO: DeviceOp<Output = T>> DeviceFuture<T, DO> {
    /// Creates an idle device future with no operation or execution context set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a device future scheduled on the given stream.
    pub fn scheduled(op: DO, ctx: ExecutionContext) -> Self {
        Self {
            device_operation: Some(op),
            execution_context: Some(ctx),
            ..Default::default()
        }
    }

    /// Create a future that is pre-loaded with an error.
    ///
    /// On first poll it immediately returns `Poll::Ready(Err(error))`.
    /// This is used by `IntoFuture` implementations to surface scheduling
    /// failures without panicking.
    pub fn failed(error: DeviceError) -> Self {
        Self {
            execution_context: None,
            device_operation: None,
            state: DeviceFutureState::Failed,
            callback_state: None,
            result: None,
            error: Some(error),
        }
    }

    /// Registers a host callback on the CUDA stream to signal completion.
    ///
    /// # Safety
    /// The execution context's stream must be valid for the lifetime of the callback.
    unsafe fn register_callback(
        &self,
        waker_state: Arc<StreamCallbackState>,
    ) -> Result<(), DeviceError> {
        let ctx = self
            .execution_context
            .as_ref()
            .ok_or(DeviceError::Internal(
                "Cannot execute future without setting stream on which to execute.".to_string(),
            ))?;
        ctx.get_cuda_stream().launch_host_function(move || {
            waker_state.signal();
        })?;
        Ok(())
    }
    /// Executes the stored device operation on the associated stream.
    fn execute(&mut self) -> Result<(), DeviceError> {
        let ctx = self
            .execution_context
            .as_ref()
            .ok_or(DeviceError::Internal(
                "Cannot execute future without setting stream on which to execute.".to_string(),
            ))?;
        // TODO (hme): We may need to hold a reference to device_operation,
        //  to ensure kernel launch structs (and their args) are dropped
        //  when the future completes vs. when this function completes.
        let operation = self.device_operation.take().ok_or(DeviceError::Internal(
            "Unable to execute future: No operation has been set.".to_string(),
        ))?;
        let out = unsafe { operation.execute(ctx) }?;
        self.result = Some(out);
        Ok(())
    }
}

impl<T: Send, DO: DeviceOp<Output = T>> Default for DeviceFuture<T, DO> {
    fn default() -> Self {
        Self {
            device_operation: None,
            execution_context: None,
            result: None,
            error: None,
            state: DeviceFutureState::Idle,
            callback_state: None,
        }
    }
}

impl<T: Send, DO: DeviceOp<Output = T>> Unpin for DeviceFuture<T, DO> {}

impl<T: Send, DO: DeviceOp<Output = T>> Future for DeviceFuture<T, DO> {
    type Output = Result<T, DeviceError>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.state == DeviceFutureState::Failed {
            self.state = DeviceFutureState::Complete;
            let error = self
                .error
                .take()
                .expect("Failed state must carry an error.");
            return Poll::Ready(Err(error));
        }

        // If this is being polled, it needs a waker.
        if self.callback_state.is_none() {
            self.callback_state = Some(Arc::new(StreamCallbackState::new()));
        }
        let waker_state = self.callback_state.as_ref().cloned().expect("Impossible.");
        match self.state {
            DeviceFutureState::Idle => {
                // Acquire the thread-local execution lock.
                if let Err(e) = crate::device_operation::acquire_execution_lock() {
                    self.state = DeviceFutureState::Complete;
                    return Poll::Ready(Err(e));
                }
                // Initialize the waker.
                waker_state.waker.register(cx.waker());
                // Execute this future's operation.
                if let Err(e) = self.execute() {
                    crate::device_operation::release_execution_lock();
                    self.state = DeviceFutureState::Complete;
                    return Poll::Ready(Err(e));
                }
                // Add the callback. We only want to do this once.
                if let Err(e) = unsafe { self.register_callback(waker_state.clone()) } {
                    crate::device_operation::release_execution_lock();
                    self.state = DeviceFutureState::Complete;
                    return Poll::Ready(Err(e));
                }
                // Transition the future's state to "Executing."
                // Release the lock — the GPU work is submitted and the
                // callback will signal completion asynchronously.
                crate::device_operation::release_execution_lock();
                self.state = DeviceFutureState::Executing;
                Poll::Pending
            }
            DeviceFutureState::Executing => {
                // The future may have been polled by the waker firing or by some other mechanism.
                // Check if the complete flag has been set by the callback.
                if waker_state.complete.load(Ordering::Relaxed) {
                    self.state = DeviceFutureState::Complete;
                    // If the future was polled by some mechanism other than the waker,
                    // then the old waker still may fire, but the future will not be polled
                    // again if we return Poll::Ready.
                    return Poll::Ready(Ok(self
                        .result
                        .take()
                        .expect("Expected future result to be Some.")));
                }
                // The future is still incomplete. Update the waker to the latest context.
                waker_state.waker.register(cx.waker());
                // Check if the callback has fired after updating the waker.
                // If the callback triggers the old waker before the new waker is registered,
                // the newly registered waker will never be called.
                if waker_state.complete.load(Ordering::Relaxed) {
                    self.state = DeviceFutureState::Complete;
                    Poll::Ready(Ok(self
                        .result
                        .take()
                        .expect("Expected future result to be Some.")))
                } else {
                    Poll::Pending
                }
            }
            DeviceFutureState::Complete => {
                // We set the future's state to complete before returning Poll::Ready.
                // The executor *should* never poll this task again.
                panic!("Poll called after completion.");
            }
            DeviceFutureState::Failed => {
                // Already handled above; this arm is unreachable.
                unreachable!();
            }
        }
    }
}
