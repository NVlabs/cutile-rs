// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

use std::{env, error::Error, fs, path::Path, process::exit};

use bindgen::CodegenConfig;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Expr, Fields, File, GenericArgument, Item, ItemStruct, PathArguments, ReturnType, Type,
    TypeBareFn,
};

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
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_PATH");

    let cuda_toolkit =
        env::var("CUDA_TOOLKIT_PATH").expect("CUDA_TOOLKIT_PATH is required but not set");
    let out_dir = env::var("OUT_DIR")?;
    let out_dir = Path::new(&out_dir);

    generate_type_bindings(&cuda_toolkit, out_dir)?;
    for spec in API_SPECS {
        generate_dynamic_api(&cuda_toolkit, out_dir, spec)?;
    }

    Ok(())
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
