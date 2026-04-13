/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Error types and helpers for macro diagnostics.
//!
//! Provides span-aware error construction for producing compile-time error messages.

/// Unified error type for proc-macro diagnostics.
#[derive(Debug)]
pub enum Error {
    /// Wraps a `syn::Error` with span information.
    Syn(syn::Error),
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Syn(error) => write!(formatter, "Syntax error: {error}"),
        }
    }
}

impl<T> From<Error> for Result<T, Error> {
    #[inline]
    fn from(value: Error) -> Self {
        Err(value)
    }
}

/// Extension trait for producing span-anchored errors from any `Spanned` item.
pub trait SpannedError {
    /// Return `Error` anchored to this item's span.
    fn error(&self, message: &str) -> Error;
    /// Return `Err(Error)` anchored to this item's span, with an arbitrary `Ok` type.
    fn err<T>(&self, message: &str) -> Result<T, Error>;
}

impl<S> SpannedError for S
where
    S: syn::spanned::Spanned,
{
    fn error(&self, message: &str) -> Error {
        Error::Syn(syn::Error::new(self.span(), message))
    }
    #[inline]
    fn err<T>(&self, message: &str) -> Result<T, Error> {
        self.error(message).into()
    }
}
