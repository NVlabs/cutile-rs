/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unit tests for `FileSystemJitStore`.
//! These tests run on CPU — no GPU required.

use cuda_async::jit_store::{FileSystemJitStore, JitStore};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Monotonically increasing counter so each test gets its own directory,
/// even when tests run in parallel within the same process.
static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn tmp_store() -> (FileSystemJitStore, PathBuf) {
    let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("cutile_jit_test_{}_{id}", std::process::id()));
    let store = FileSystemJitStore::new(dir.clone()).expect("failed to create store");
    (store, dir)
}

fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn put_and_get() {
    let (store, dir) = tmp_store();
    let data = b"fake cubin data";
    store.put("abc123", data).unwrap();
    let result = store.get("abc123").unwrap();
    assert_eq!(result, Some(data.to_vec()));
    cleanup(&dir);
}

#[test]
fn get_missing_returns_none() {
    let (store, dir) = tmp_store();
    let result = store.get("nonexistent").unwrap();
    assert_eq!(result, None);
    cleanup(&dir);
}

#[test]
fn contains() {
    let (store, dir) = tmp_store();
    assert!(!store.contains("key1").unwrap());
    store.put("key1", b"data").unwrap();
    assert!(store.contains("key1").unwrap());
    cleanup(&dir);
}

#[test]
fn delete() {
    let (store, dir) = tmp_store();
    store.put("key2", b"data").unwrap();
    assert!(store.contains("key2").unwrap());
    store.delete("key2").unwrap();
    assert!(!store.contains("key2").unwrap());
    // Deleting a nonexistent key is a no-op.
    store.delete("key2").unwrap();
    cleanup(&dir);
}

#[test]
fn clear() {
    let (store, dir) = tmp_store();
    store.put("a", b"1").unwrap();
    store.put("b", b"2").unwrap();
    store.clear().unwrap();
    assert!(!store.contains("a").unwrap());
    assert!(!store.contains("b").unwrap());
    cleanup(&dir);
}

#[test]
fn put_overwrites() {
    let (store, dir) = tmp_store();
    store.put("key", b"old").unwrap();
    store.put("key", b"new").unwrap();
    let result = store.get("key").unwrap();
    assert_eq!(result, Some(b"new".to_vec()));
    cleanup(&dir);
}

#[test]
fn large_data() {
    let (store, dir) = tmp_store();
    let data = vec![0xABu8; 1024 * 1024]; // 1 MB
    store.put("large", &data).unwrap();
    let result = store.get("large").unwrap().unwrap();
    assert_eq!(result.len(), data.len());
    assert_eq!(result, data);
    cleanup(&dir);
}

#[test]
fn concurrent_put_get() {
    use std::sync::Arc;
    let (store, dir) = tmp_store();
    let store = Arc::new(store);
    let mut handles = vec![];
    for i in 0..8 {
        let store = Arc::clone(&store);
        handles.push(std::thread::spawn(move || {
            let key = format!("concurrent_{i}");
            let data = vec![i as u8; 256];
            store.put(&key, &data).unwrap();
            let result = store.get(&key).unwrap().unwrap();
            assert_eq!(result, data);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    cleanup(&dir);
}

#[test]
fn clear_on_empty_dir() {
    let (store, dir) = tmp_store();
    // Clear on an empty store should succeed.
    store.clear().unwrap();
    store.clear().unwrap();
    cleanup(&dir);
}

#[test]
fn keys_with_hex_characters() {
    let (store, dir) = tmp_store();
    // Realistic SHA-256 hex key.
    let key = "a3f8b2c1deadbeef0123456789abcdef0123456789abcdef0123456789abcdef";
    store.put(key, b"cubin bytes").unwrap();
    assert!(store.contains(key).unwrap());
    assert_eq!(store.get(key).unwrap(), Some(b"cubin bytes".to_vec()));
    cleanup(&dir);
}
