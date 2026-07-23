/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU tests for the on-disk cubin cache.
//!
//! The cross-process test is the key acceptance condition: a kernel compiled
//! once must be served from disk to a *fresh process* without spawning
//! `tileiras`.
//!
//! Each test uses a distinct tile size so it is an in-memory cache miss even
//! when the other tests in this binary already compiled the kernel — only an
//! L1 miss reaches the disk layer.

use crate::common;
use cutile::api::{ones, zeros};
use cutile::jit_cache::{self, jit_backend_compile_count, jit_disk_hit_count, FileSystemJitStore};
use cutile::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[cutile::module]
mod jit_disk_cache_test_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }
}

/// Launches `add` with the given tile size and asserts the numeric result, so
/// a disk-served cubin is verified end to end, not just loaded.
fn launch_and_check(tile: usize) {
    let x: Arc<Tensor<f32>> = ones(&[256]).sync().expect("ones").into();
    let y: Arc<Tensor<f32>> = ones(&[256]).sync().expect("ones").into();
    let z = zeros(&[256]).sync().expect("zeros").partition([tile]);
    let z_host = jit_disk_cache_test_module::add(z, x, y)
        .unzip()
        .0
        .unpartition()
        .to_host_vec()
        .sync()
        .expect("add kernel");
    assert!(z_host.iter().all(|&v| (v - 2.0f32).abs() < 1e-6));
}

fn fresh_dir(label: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "cutile_disk_cache_gpu_{label}_{}_{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed),
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn enable_at(dir: &Path) {
    jit_cache::enable(Arc::new(FileSystemJitStore::new(dir).expect("open store")));
}

/// Env vars the orchestrator uses to drive its child processes. Read only by
/// this test — the library itself has no env switch.
const CHILD_DIR_ENV: &str = "CUTILE_TEST_JIT_DISK_CACHE_CHILD_DIR";
const CHILD_ROLE_ENV: &str = "CUTILE_TEST_JIT_DISK_CACHE_ROLE";

/// The cross-process acceptance condition: a kernel compiled by one process is served from
/// disk to another, with no `tileiras` spawn in the second.
///
/// Both the writer and the reader run in **fresh** processes, so their
/// in-memory kernel cache is empty and every kernel reaches the disk layer.
/// The orchestrator cannot play the writer itself: it shares this test
/// binary's in-memory cache, which earlier tests have already warmed for the
/// `ones`/`zeros`/`to_host_vec` helper kernels, so those would never reach the
/// disk and the reader would then miss them.
#[test]
fn disk_cache_cross_process_hit() {
    if let (Some(dir), Some(role)) = (
        std::env::var_os(CHILD_DIR_ENV),
        std::env::var_os(CHILD_ROLE_ENV),
    ) {
        let is_reader = role.to_str() == Some("reader");
        common::with_test_stack(move || {
            enable_at(Path::new(&dir));
            launch_and_check(64);
            jit_cache::disable();
            if is_reader {
                assert_eq!(
                    jit_backend_compile_count(),
                    0,
                    "reader must not spawn tileiras: the writer stored every kernel it launches"
                );
                assert!(
                    jit_disk_hit_count() >= 1,
                    "reader must serve kernels from the disk cache, got {}",
                    jit_disk_hit_count(),
                );
            } else {
                assert_eq!(
                    jit_disk_hit_count(),
                    0,
                    "writer starts cold: nothing is on disk yet"
                );
                assert!(
                    jit_backend_compile_count() >= 1,
                    "writer must compile and store at least one kernel"
                );
            }
        });
        return;
    }

    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let dir = fresh_dir("xproc");
        let exe = std::env::current_exe().expect("current_exe");

        let run = |role: &str| {
            std::process::Command::new(&exe)
                .args([
                    "jit_disk_cache::disk_cache_cross_process_hit",
                    "--exact",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_DIR_ENV, &dir)
                .env(CHILD_ROLE_ENV, role)
                .output()
                .expect("spawn child test process")
        };

        for role in ["writer", "reader"] {
            let out = run(role);
            assert!(
                out.status.success(),
                "{role} process failed.\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    });
}

/// Regression: with no store enabled, a compile writes nothing to disk.
#[test]
fn disk_cache_default_off_writes_nothing() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        jit_cache::disable();
        assert!(!jit_cache::is_enabled());

        let puts_before = jit_cache::stats().puts;
        launch_and_check(32);
        assert_eq!(
            jit_cache::stats().puts,
            puts_before,
            "no store is installed, so nothing may be written"
        );
    });
}

/// A store that can neither read nor write must not fail the launch — the
/// errors are counted and the compile proceeds.
#[test]
fn disk_cache_degrades_softly_on_io_errors() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let dir = fresh_dir("degraded");
        let root = dir.join("store");
        enable_at(&root); // creates `root` as a directory

        // Replace the store root with a regular file. Every store operation
        // then fails with ENOTDIR (`<root>/<shard>/...` traverses a file),
        // which even root cannot bypass — directory permission bits, by
        // contrast, are ignored for root, and this suite runs as root.
        std::fs::remove_dir_all(&root).unwrap();
        std::fs::write(&root, b"not a directory").unwrap();

        let io_errors_before = jit_cache::stats().io_errors;
        launch_and_check(128); // … must still succeed
        jit_cache::disable();
        assert!(
            jit_cache::stats().io_errors > io_errors_before,
            "the broken store must surface as counted soft errors"
        );

        let _ = std::fs::remove_dir_all(&dir);
    });
}
