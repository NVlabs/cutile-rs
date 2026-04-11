/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Object-store-like interface for persisting JIT compilation artifacts to disk.
//!
//! The [`JitStore`] trait provides a simple key-value interface for caching compiled
//! cubins. Keys are SHA-256 hashes that encode all factors affecting compilation output
//! (source content, GPU architecture, compiler version, toolkit version).
//!
//! [`FileSystemJitStore`] is the default implementation, storing artifacts as individual
//! files under a configurable directory (defaults to `~/.cache/cutile/`).

use std::io;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Object-store-like interface for persisting JIT compilation artifacts.
///
/// Implementations store compiled cubins keyed by a SHA-256 hash string.
/// The hash encodes all factors that affect compilation output:
/// source content, GPU architecture, compiler version, and toolkit version.
pub trait JitStore: Send + Sync {
    /// Retrieve a cached artifact by key.
    fn get(&self, key: &str) -> io::Result<Option<Vec<u8>>>;

    /// Store a compiled artifact.
    fn put(&self, key: &str, data: &[u8]) -> io::Result<()>;

    /// Check whether an artifact exists without reading it.
    fn contains(&self, key: &str) -> io::Result<bool>;

    /// Remove a cached artifact.
    fn delete(&self, key: &str) -> io::Result<()>;

    /// Remove all cached artifacts.
    fn clear(&self) -> io::Result<()>;
}

/// Filesystem-backed JIT artifact store.
///
/// Stores compiled cubins as individual `.cubin` files under a base directory.
pub struct FileSystemJitStore {
    base_dir: PathBuf,
}

impl FileSystemJitStore {
    /// Create a new store at the given directory, creating it if necessary.
    pub fn new(base_dir: PathBuf) -> io::Result<Self> {
        std::fs::create_dir_all(&base_dir)?;
        Ok(Self { base_dir })
    }

    /// Create a store at the default location (`~/.cache/cutile/`).
    pub fn default_location() -> io::Result<Self> {
        let dir = dirs_default_cache_dir().join("cutile");
        Self::new(dir)
    }

    fn artifact_path(&self, key: &str) -> PathBuf {
        self.base_dir.join(format!("{key}.cubin"))
    }
}

/// Returns a default cache directory, similar to `dirs::cache_dir()`.
fn dirs_default_cache_dir() -> PathBuf {
  
    #[cfg(target_os = "linux")]
    {
        if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
            return PathBuf::from(xdg);
        }
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(".cache");
        }
    }
    PathBuf::from("/tmp")
}

impl JitStore for FileSystemJitStore {
    fn get(&self, key: &str) -> io::Result<Option<Vec<u8>>> {
        let path = self.artifact_path(key);
        match std::fs::read(&path) {
            Ok(data) => Ok(Some(data)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn put(&self, key: &str, data: &[u8]) -> io::Result<()> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = self.artifact_path(key);
        // Write to a uniquely-named temp file, then rename for atomicity.
        // The PID + counter suffix prevents collisions across threads and processes.
        let tmp_path = path.with_extension(format!("cubin.tmp.{}.{}", std::process::id(), n));
        std::fs::write(&tmp_path, data)?;
        std::fs::rename(&tmp_path, &path)
    }

    fn contains(&self, key: &str) -> io::Result<bool> {
        Ok(self.artifact_path(key).exists())
    }

    fn delete(&self, key: &str) -> io::Result<()> {
        let path = self.artifact_path(key);
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn clear(&self) -> io::Result<()> {
        if self.base_dir.exists() {
            std::fs::remove_dir_all(&self.base_dir)?;
            std::fs::create_dir_all(&self.base_dir)?;
        }
        Ok(())
    }
}

// ── Global JitStore configuration ───────────────────────────────────────────

static JIT_STORE: OnceLock<Option<Box<dyn JitStore>>> = OnceLock::new();

/// Configure the global JIT store. Call once at startup.
///
/// Pass `None` to disable disk persistence. Panics if called more than once.
pub fn set_jit_store(store: Option<Box<dyn JitStore>>) {
    if JIT_STORE.set(store).is_err() {
        panic!("JIT store has already been configured");
    }
}

/// Try to configure the global JIT store. Returns `true` if successfully set,
/// `false` if a store was already configured (in which case the argument is dropped).
///
/// This is useful in test code where multiple tests may race to set the store.
pub fn set_jit_store_if_unset(store: Option<Box<dyn JitStore>>) -> bool {
    JIT_STORE.set(store).is_ok()
}

/// Get a reference to the global JIT store, if one has been configured.
pub fn get_jit_store() -> Option<&'static dyn JitStore> {
    JIT_STORE.get().and_then(|s| s.as_ref().map(|b| b.as_ref()))
}

/// Ensure a JIT disk cache is configured.
///
/// If no store has been explicitly set via [`set_jit_store`] or
/// [`set_jit_store_if_unset`], this lazily initializes a
/// [`FileSystemJitStore`] at the default location (`~/.cache/cutile/` on Linux).
/// Set `CUTILE_NO_DISK_CACHE=1` to disable auto-initialization.
///
/// This is called automatically from the compilation pipeline.  Users who want
/// a custom store should call [`set_jit_store`] *before* any kernel compilation.
pub fn ensure_default_jit_store() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        if std::env::var("CUTILE_NO_DISK_CACHE").is_ok_and(|v| v == "1") {
            return;
        }
        if let Ok(store) = FileSystemJitStore::default_location() {
            // Best-effort: if set_jit_store was already called, this is a no-op.
            let _ = set_jit_store_if_unset(Some(Box::new(store)));
        }
    });
}
