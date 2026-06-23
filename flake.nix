{
  description = "Development shell for cutile-rs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        isDarwin = pkgs.stdenv.isDarwin;

        # CUDA 13.3 fetched from nvidia (headers needed on all platforms for
        # bindgen type generation; runtime libraries only needed on Linux)
        cudaRedistBase = "https://developer.download.nvidia.com/compute/cuda/redist";
        fetchCudaRedist =
          { name, version, sha256 }:
          pkgs.fetchurl {
            url = "${cudaRedistBase}/${name}/linux-x86_64/${name}-linux-x86_64-${version}-archive.tar.xz";
            inherit sha256;
          };
        cudaToolkit = pkgs.symlinkJoin {
          name = "cuda-toolkit-13.3";
          paths = map (src: pkgs.runCommand (builtins.baseNameOf (toString src)) { } ''
            mkdir -p $out
            tar xf ${src} --strip-components=1 -C $out
          '') [
            (fetchCudaRedist {
              name = "cuda_crt";
              version = "13.3.33";
              sha256 = "4755d36d24c6ef7697a2d3e1dbb23c4562c9c0d97d48390d4cbd8ab32dec5b5f";
            })
            (fetchCudaRedist {
              name = "cuda_nvcc";
              version = "13.3.33";
              sha256 = "93b098bda4a562ebf3541523ce82adc43f106a81dcf28bcbf8f0d8e093d1c66f";
            })
            (fetchCudaRedist {
              name = "libnvvm";
              version = "13.3.33";
              sha256 = "fc9c1fd5844e44c0e5eeb051378c1b13cf0e3bb3fe4966d5103c38885424f802";
            })
            (fetchCudaRedist {
              name = "cuda_cudart";
              version = "13.3.29";
              sha256 = "1e59c4888267d27ba1a9bd0f3669a6439db1334a96e754cd9013c7c73e18dc9d";
            })
            (fetchCudaRedist {
              name = "libcurand";
              version = "10.4.3.29";
              sha256 = "0218e62ab413e435dcd0274ec8e63b62214e6aba8519201061d1597e73caadbb";
            })
            (fetchCudaRedist {
              name = "cuda_tileiras";
              version = "13.3.36";
              sha256 = "1b055db199f806c746d53331200ccd8480bfdddd14638ed2911f30ee0cc4447b";
            })
          ];
          postBuild = ''
            # cuda-bindings/build.rs also looks for libs in lib64/
            if [ -d "$out/lib" ] && [ ! -e "$out/lib64" ]; then
              ln -s lib "$out/lib64"
            fi
          '';
        };

        # bindgen uses libclang to parse CUDA headers.
        libclang = pkgs.llvmPackages.libclang;
        libcDev = pkgs.lib.getDev pkgs.stdenv.cc.libc;
        bindgenClangArgs = pkgs.lib.optionalString (!isDarwin) "-isystem ${libcDev}/include";

        # Nightly Rust 
        rustToolchain =
          pkgs.rust-bin.nightly."2025-07-16".default.override
            {
              extensions = [
                "clippy"
                "rust-analyzer"
                "rust-src"
                "rustfmt"
              ];
            };
      in
      {
        devShells.default = pkgs.mkShell {
          hardeningDisable = [ "fortify" ];

          packages = [
            rustToolchain
            libclang
            pkgs.cmake
            pkgs.git
            pkgs.libffi
            pkgs.libxml2
            pkgs.ninja
            pkgs.pkg-config
            pkgs.python3
            pkgs.which
          ];

          CMAKE_GENERATOR = "Ninja";
          CUDA_TOOLKIT_PATH = "${cudaToolkit}";
          LIBCLANG_PATH = "${libclang.lib}/lib";
          BINDGEN_EXTRA_CLANG_ARGS = bindgenClangArgs;

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath ([
            pkgs.libffi
            pkgs.libxml2
            libclang.lib
          ] ++ pkgs.lib.optionals (!isDarwin) [ cudaToolkit ]);

          shellHook = pkgs.lib.optionalString (!isDarwin) ''
            export PATH="${cudaToolkit}/bin:$PATH"
            # GPU driver libs: NixOS provides /run/opengl-driver/lib; on other
            # distros, symlink just the NVIDIA libs into a temp dir so we don't
            # pull in the host glibc.
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            else
              _nv_drv_dir=$(mktemp -d /tmp/nix-nvidia-driver.XXXXXX)
              for d in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu /usr/lib /usr/lib64; do
                if [ -e "$d/libcuda.so.1" ]; then
                  for lib in "$d"/libcuda.so* "$d"/libnvidia-ptxjitcompiler.so* "$d"/libnvidia-gpucomp.so*; do
                    [ -e "$lib" ] && ln -sf "$lib" "$_nv_drv_dir/"
                  done
                  break
                fi
              done
              if [ -n "$(ls -A "$_nv_drv_dir" 2>/dev/null)" ]; then
                export LD_LIBRARY_PATH="$_nv_drv_dir:$LD_LIBRARY_PATH"
              else
                rm -rf "$_nv_drv_dir"
              fi
            fi
          '' + ''
            echo ""
            echo "cutile-rs dev shell"
            echo " ${if isDarwin then "~" else "✓"} CUDA  ${if isDarwin then "(headers only — no GPU required)" else "$CUDA_TOOLKIT_PATH"}"
            echo " ✓ Rust  $(rustc --version 2>/dev/null | awk '{print $2}')"
          '' + pkgs.lib.optionalString isDarwin ''
            echo ""
            echo "macOS: compile-only mode. Run:"
            echo "  cargo run -p cutile-examples --example compile_only"
          '' + ''
            echo ""
          '';
        };
      }
    );
}
