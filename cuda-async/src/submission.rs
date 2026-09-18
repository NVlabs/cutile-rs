/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Owners held independently of an operation's (possibly borrowed) output.

use crate::device_future::{probe_stream, StreamHealth};
use crate::device_operation::{ExecutionContext, ReplayResource};
use crate::error::DeviceError;
use cuda_core::Stream;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Owners(Option<Vec<Box<dyn Send>>>);

impl Owners {
    fn retain(&mut self, owner: impl Send + 'static) -> Result<(), DeviceError> {
        self.0
            .as_mut()
            .ok_or_else(|| DeviceError::Internal("submission already completed".into()))?
            .push(Box::new(owner));
        Ok(())
    }

    fn release(&mut self, wait: impl FnOnce() -> Result<(), DeviceError>) {
        let Some(owners) = self.0.take() else { return };
        if !owners.is_empty() && wait().is_err() {
            // This includes access leases, not just allocations. Unblocking an
            // access when completion is unknown would permit a device data race.
            std::mem::forget(owners);
        }
    }
}

pub(crate) struct Submission {
    stream: Arc<Stream>,
    owners: Mutex<Owners>,
    recorded: Mutex<Vec<Arc<dyn ReplayResource>>>,
}

impl std::fmt::Debug for Submission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Submission").finish_non_exhaustive()
    }
}

impl Submission {
    pub(crate) fn new(stream: Arc<Stream>) -> Self {
        Self {
            stream,
            owners: Mutex::new(Owners(Some(Vec::new()))),
            recorded: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn retain(&self, owner: impl Send + 'static) -> Result<(), DeviceError> {
        self.owners
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(owner)
    }

    pub(crate) fn record(&self, resource: Arc<dyn ReplayResource>) {
        self.recorded
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(resource);
    }

    pub(crate) fn replay(&self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        for resource in self
            .recorded
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
        {
            resource.retain_for_launch(ctx)?;
        }
        Ok(())
    }

    /// No further work may be submitted using this submission.
    pub(crate) unsafe fn complete(&self) {
        let owners = self
            .owners
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .0
            .take();
        drop(owners);
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        self.owners
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .release(|| {
                let result = (|| {
                    self.stream.device().bind_to_thread()?;
                    match probe_stream(&self.stream) {
                        StreamHealth::Idle => Ok(()),
                        StreamHealth::Busy => {
                            unsafe { self.stream.synchronize() }.map_err(DeviceError::Driver)
                        }
                        StreamHealth::Faulted(e) => Err(DeviceError::Driver(e)),
                        StreamHealth::Capturing => Err(DeviceError::Internal(
                            "submission is still being captured".into(),
                        )),
                    }
                })();
                if result.is_err() {
                    // Keep the stream identity valid for leaked access leases too.
                    std::mem::forget(self.stream.clone());
                }
                result
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owners_outlive_recovered_or_projected_results() {
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        let mut submission = Owners(Some(Vec::new()));
        submission.retain(owner).unwrap();
        assert!(weak.upgrade().is_some());
        submission.release(|| {
            assert!(weak.upgrade().is_some());
            Ok(())
        });
        assert!(weak.upgrade().is_none());
        assert!(submission.retain(()).is_err());
    }

    #[test]
    fn forgotten_submission_keeps_owners() {
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        let mut submission = Owners(Some(Vec::new()));
        submission.retain(owner).unwrap();
        std::mem::forget(submission);
        assert!(weak.upgrade().is_some());
    }

    #[test]
    fn failed_wait_keeps_owners_and_access_leases() {
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        let mut submission = Owners(Some(Vec::new()));
        submission.retain(owner).unwrap();
        submission.release(|| Err(DeviceError::Internal("fault".into())));
        assert!(weak.upgrade().is_some());
    }
}
