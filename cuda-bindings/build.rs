// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

use std::{
    env,
    error::Error,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::exit,
};

use bindgen::CodegenConfig;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Expr, Fields, File, GenericArgument, Item, ItemStruct, PathArguments, ReturnType, Type,
    TypeBareFn,
};

const MIN_CUDA_VERSION: u32 = 13020;
const CUDA_TOOLKIT_PATH_ENV: &str = "CUDA_TOOLKIT_PATH";
const SETUP_DIAGNOSTICS_ENV: &str = "CUTILE_SETUP_DIAGNOSTICS";

struct ApiSpec {
    virtual_header: &'static str,
    header_contents: &'static str,
    generated_api: &'static str,
    generated_shims: &'static str,
    library_name: &'static str,
    api_type: &'static str,
    loader_fn: &'static str,
    fallback_expr: &'static str,
    function_pattern: &'static str,
}

const API_SPECS: &[ApiSpec] = &[
    ApiSpec {
        virtual_header: "cuda_driver_wrapper.h",
        header_contents: "#include <cuda.h>\n",
        generated_api: "cuda_driver_api.rs",
        generated_shims: "cuda_driver_shims.rs",
        library_name: "CudaDriverApi",
        api_type: "CudaDriverApi",
        loader_fn: "cuda_driver",
        fallback_expr: "cudaError_enum_CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
        function_pattern: "^cu.*",
    },
    ApiSpec {
        virtual_header: "curand_wrapper.h",
        header_contents: "#include <curand.h>\n",
        generated_api: "curand_api.rs",
        generated_shims: "curand_shims.rs",
        library_name: "CurandApi",
        api_type: "CurandApi",
        loader_fn: "curand_api",
        fallback_expr: "curandStatus_CURAND_STATUS_NOT_INITIALIZED",
        function_pattern: "^curand.*",
    },
];

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed={CUDA_TOOLKIT_PATH_ENV}");
    println!("cargo:rerun-if-env-changed={SETUP_DIAGNOSTICS_ENV}");

    let cuda_toolkit = resolve_cuda_toolkit()?;
    let cuda_toolkit = cuda_toolkit.to_string_lossy().into_owned();
    println!("cargo:rustc-env=CUTILE_RESOLVED_CUDA_TOOLKIT_PATH={cuda_toolkit}");
    let out_dir = env::var("OUT_DIR")?;
    let out_dir = Path::new(&out_dir);

    generate_type_bindings(&cuda_toolkit, out_dir)?;
    for spec in API_SPECS {
        generate_dynamic_api(&cuda_toolkit, out_dir, spec)?;
    }

    Ok(())
}

fn resolve_cuda_toolkit() -> Result<PathBuf, Box<dyn Error>> {
    match env::var_os(CUDA_TOOLKIT_PATH_ENV) {
        Some(value) if value.is_empty() => Err(format!(
            "{CUDA_TOOLKIT_PATH_ENV} is set to an empty string. Set it to a CUDA 13.2+ toolkit directory."
        )
        .into()),
        Some(value) => resolve_explicit_cuda_toolkit(value),
        None => find_default_cuda_toolkit(),
    }
}

fn resolve_explicit_cuda_toolkit(value: OsString) -> Result<PathBuf, Box<dyn Error>> {
    let cuda_toolkit = PathBuf::from(value);
    let version = validate_cuda_toolkit(&cuda_toolkit).map_err(|error| {
        format!(
            "{CUDA_TOOLKIT_PATH_ENV}={} is invalid: {error}",
            cuda_toolkit.display()
        )
    })?;
    emit_setup_diagnostic(format_args!(
        "using {CUDA_TOOLKIT_PATH_ENV}={} (CUDA {})",
        cuda_toolkit.display(),
        format_cuda_version(version)
    ));
    Ok(cuda_toolkit)
}

fn find_default_cuda_toolkit() -> Result<PathBuf, Box<dyn Error>> {
    let mut rejected = Vec::new();
    for cuda_toolkit in default_cuda_toolkit_candidates() {
        match validate_cuda_toolkit(cuda_toolkit) {
            Ok(version) => {
                emit_setup_diagnostic(format_args!(
                    "{CUDA_TOOLKIT_PATH_ENV} is unset; using discovered CUDA toolkit {} (CUDA {})",
                    cuda_toolkit.display(),
                    format_cuda_version(version)
                ));
                return Ok(cuda_toolkit.to_path_buf());
            }
            Err(error) => {
                emit_setup_diagnostic(format_args!(
                    "{CUDA_TOOLKIT_PATH_ENV} is unset; skipping {}: {error}",
                    cuda_toolkit.display()
                ));
                rejected.push(format!("  - {}: {error}", cuda_toolkit.display()));
            }
        }
    }

    Err(format!(
        "{CUDA_TOOLKIT_PATH_ENV} is not set, and no CUDA 13.2+ toolkit was found in default locations:\n{}\nSet {CUDA_TOOLKIT_PATH_ENV} to a CUDA 13.2+ toolkit directory.",
        rejected.join("\n")
    )
    .into())
}

fn default_cuda_toolkit_candidates() -> &'static [PathBuf] {
    static CANDIDATES: std::sync::OnceLock<Vec<PathBuf>> = std::sync::OnceLock::new();
    CANDIDATES.get_or_init(|| {
        #[cfg(windows)]
        let candidates = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
        ];
        #[cfg(not(windows))]
        let candidates = [
            "/usr/local/cuda-13.3",
            "/usr/local/cuda-13.2",
            "/usr/local/cuda-13",
            "/usr/local/cuda",
        ];

        candidates.into_iter().map(PathBuf::from).collect()
    })
}

fn validate_cuda_toolkit(cuda_toolkit: &Path) -> Result<u32, String> {
    if !cuda_toolkit.is_dir() {
        return Err(format!("{} is not a directory", cuda_toolkit.display()));
    }

    let cuda_h = cuda_toolkit.join("include").join("cuda.h");
    if !cuda_h.is_file() {
        return Err(format!(
            "{} does not contain include/cuda.h",
            cuda_toolkit.display(),
        ));
    }

    let version = cuda_version_from_header(&cuda_h).map_err(|error| error.to_string())?;
    if version < MIN_CUDA_VERSION {
        return Err(format!(
            "CUDA toolkit {} is too old. cuTile requires CUDA 13.2+ and recommends CUDA 13.3",
            format_cuda_version(version)
        ));
    }

    Ok(version)
}

fn cuda_version_from_header(cuda_h: &Path) -> Result<u32, Box<dyn Error>> {
    let source = fs::read_to_string(cuda_h)?;
    source
        .lines()
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            match (parts.next(), parts.next(), parts.next()) {
                (Some("#define"), Some("CUDA_VERSION"), Some(version)) => version.parse().ok(),
                _ => None,
            }
        })
        .ok_or_else(|| {
            format!(
                "could not find CUDA_VERSION in {}. Set CUDA_TOOLKIT_PATH to a CUDA 13.2+ toolkit directory.",
                cuda_h.display()
            )
            .into()
        })
}

fn format_cuda_version(version: u32) -> String {
    format!("{}.{}", version / 1000, (version % 1000) / 10)
}

fn setup_diagnostics_enabled() -> bool {
    env::var(SETUP_DIAGNOSTICS_ENV)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn emit_setup_diagnostic(args: std::fmt::Arguments<'_>) {
    if setup_diagnostics_enabled() {
        println!("cargo:warning={args}");
    }
}

fn generate_type_bindings(cuda_toolkit: &str, out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let bindings = bindgen_builder(cuda_toolkit)
        .header("wrapper.h")
        .blocklist_function(".*")
        .generate()?;
    let source = bindings.to_string();
    emit_layout_cfgs(&source);
    fs::write(out_dir.join("types.rs"), source)?;
    Ok(())
}

// Detect whether bindgen wrapped `CUmemLocation_st::id` in an anonymous union (e.g. CUDA 13.2+)
// or left it as a plain `int` (e.g. 13.0/13.1), so the helper in lib.rs can pick the right access path.
fn emit_layout_cfgs(generated_source: &str) {
    println!("cargo:rustc-check-cfg=cfg(cu_mem_location_anon_union)");
    if cu_mem_location_uses_anon_union(generated_source) {
        println!("cargo:rustc-cfg=cu_mem_location_anon_union");
    }
}

fn cu_mem_location_uses_anon_union(generated_source: &str) -> bool {
    let Ok(file) = syn::parse_file(generated_source) else {
        return false;
    };
    file.items.iter().any(|item| match item {
        Item::Struct(item_struct) if item_struct.ident == "CUmemLocation_st" => {
            let Fields::Named(fields) = &item_struct.fields else {
                return false;
            };
            fields
                .named
                .iter()
                .any(|f| f.ident.as_ref().is_some_and(|i| i == "__bindgen_anon_1"))
        }
        _ => false,
    })
}

fn generate_dynamic_api(
    cuda_toolkit: &str,
    out_dir: &Path,
    spec: &ApiSpec,
) -> Result<(), Box<dyn Error>> {
    let bindings = bindgen_builder(cuda_toolkit)
        .header_contents(spec.virtual_header, spec.header_contents)
        .dynamic_library_name(spec.library_name)
        .dynamic_link_require_all(false)
        .allowlist_function(spec.function_pattern)
        .with_codegen_config(CodegenConfig::FUNCTIONS)
        .generate()?;

    let generated_source = bindings.to_string();
    fs::write(out_dir.join(spec.generated_api), &generated_source)?;
    generate_shims_from_generated_api(&generated_source, out_dir.join(spec.generated_shims), spec)?;
    Ok(())
}

fn bindgen_builder(cuda_toolkit: &str) -> bindgen::Builder {
    bindgen::builder()
        .clang_arg(format!("-I{cuda_toolkit}/include"))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
}

fn generate_shims_from_generated_api(
    generated_source: &str,
    output_path: impl AsRef<Path>,
    spec: &ApiSpec,
) -> Result<(), Box<dyn Error>> {
    let file = syn::parse_file(generated_source)?;
    let item_struct = find_api_struct(&file, spec.api_type).ok_or_else(|| {
        format!(
            "no generated struct found for {} in {}",
            spec.api_type, spec.generated_api
        )
    })?;

    let loader_fn = format_ident!("{}", spec.loader_fn);
    let fallback_expr = syn::parse_str::<Expr>(spec.fallback_expr)?;
    let shims = generate_field_shims(item_struct, &loader_fn, &fallback_expr);
    let shim_file = syn::parse2::<File>(quote!(#shims))?;

    fs::write(output_path, prettyplease::unparse(&shim_file))?;
    Ok(())
}

fn find_api_struct<'a>(file: &'a File, api_type: &str) -> Option<&'a ItemStruct> {
    file.items.iter().find_map(|item| match item {
        Item::Struct(item_struct) if item_struct.ident == api_type => Some(item_struct),
        _ => None,
    })
}

fn generate_field_shims(
    item_struct: &ItemStruct,
    loader_fn: &proc_macro2::Ident,
    fallback_expr: &Expr,
) -> TokenStream {
    let Fields::Named(fields) = &item_struct.fields else {
        return TokenStream::new();
    };

    let shim_fns = fields.named.iter().filter_map(|field| {
        let field_name = field.ident.as_ref()?;
        let bare_fn = result_bare_fn(&field.ty)?;
        let args = bare_fn
            .inputs
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                let name = arg
                    .name
                    .as_ref()
                    .map(|(ident, _)| ident.clone())
                    .unwrap_or_else(|| format_ident!("arg_{index}"));
                let ty = &arg.ty;
                (name.clone(), quote!(#name: #ty))
            })
            .collect::<Vec<_>>();
        let arg_names = args.iter().map(|(name, _)| name).collect::<Vec<_>>();
        let arg_defs = args.iter().map(|(_, def)| def).collect::<Vec<_>>();
        let ret = match &bare_fn.output {
            ReturnType::Default => quote!(),
            ReturnType::Type(_, ty) => quote!(-> #ty),
        };

        Some(quote! {
            #[allow(non_snake_case)]
            #[allow(clippy::missing_safety_doc)]
            #[allow(clippy::too_many_arguments)]
            #[allow(clippy::missing_inline_in_public_items)]
            #[inline]
            pub unsafe fn #field_name(#(#arg_defs),*) #ret {
                match #loader_fn() {
                    Ok(api) => match &api.#field_name {
                        Ok(loaded_fn) => unsafe { (*loaded_fn)(#(#arg_names),*) },
                        Err(_) => #fallback_expr,
                    },
                    Err(_) => #fallback_expr,
                }
            }
        })
    });

    quote! {
        #(#shim_fns)*
    }
}

fn result_bare_fn(ty: &Type) -> Option<&TypeBareFn> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Result" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let Some(GenericArgument::Type(Type::BareFn(bare_fn))) = args.args.first() else {
        return None;
    };
    Some(bare_fn)
}
