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

        # CUDA 13.2 fetched from nvidia (Linux only)
        cudaRedistBase = "https://developer.download.nvidia.com/compute/cuda/redist";
        fetchCudaRedist =
          { name, version, sha256 }:
          pkgs.fetchurl {
            url = "${cudaRedistBase}/${name}/linux-x86_64/${name}-linux-x86_64-${version}-archive.tar.xz";
            inherit sha256;
          };
        cudaToolkit = pkgs.symlinkJoin {
          name = "cuda-toolkit-13.2";
          paths = map (src: pkgs.runCommand (builtins.baseNameOf (toString src)) { } ''
            mkdir -p $out
            tar xf ${src} --strip-components=1 -C $out
          '') [
            (fetchCudaRedist {
              name = "cuda_crt";
              version = "13.2.51";
              sha256 = "fbc31fed55b7255591f3a19f575ca078827f5e6757d317d009f7ec1e69fcde4b";
            })
            (fetchCudaRedist {
              name = "cuda_nvcc";
              version = "13.2.51";
              sha256 = "706b996fefc59dc8d64d317fdf48d0aa84c4ae004eff43009dd918f40c5cc66a";
            })
            (fetchCudaRedist {
              name = "cuda_cudart";
              version = "13.2.51";
              sha256 = "539edc1056e44d319f2112e9971c6415d78d4dde04b3f6ffbd20ec808e718526";
            })
            (fetchCudaRedist {
              name = "libcurand";
              version = "10.4.2.51";
              sha256 = "a089985ac24fff42b719ab42a015c7df39cd721a3d83bfa4af9249b9fca883dc";
            })
            (fetchCudaRedist {
              name = "cuda_tileiras";
              version = "13.2.51";
              sha256 = "76cbbcc4458b6175878c3a1168521ca9ce36263e7e450ff8a1d1988e5b0bf792";
            })
          ];
          postBuild = ''
            # cuda-bindings/build.rs also looks for libs in lib64/
            if [ -d "$out/lib" ] && [ ! -e "$out/lib64" ]; then
              ln -s lib "$out/lib64"
            fi
          '';
        };

        # LLVM / MLIR 
        llvmPkgs = pkgs.llvmPackages_21;
        llvmIncludeRoot = pkgs.symlinkJoin {
          name = "llvm-mlir-21-headers";
          paths = [
            llvmPkgs.llvm.dev
            llvmPkgs.mlir.dev
          ];
        };
        llvmLibRoot = pkgs.symlinkJoin {
          name = "llvm-mlir-21-libs";
          paths = [
            llvmPkgs.llvm.lib
            llvmPkgs.mlir
          ];
        };
        llvmConfigWrapper = pkgs.writeShellScriptBin "llvm-config" ''
          real_llvm_config="${llvmPkgs.llvm.dev}/bin/llvm-config"
          merged_include="${llvmIncludeRoot}/include"
          merged_lib="${llvmLibRoot}/lib"
          real_include="$("$real_llvm_config" --includedir)"

          for arg in "$@"; do
            case "$arg" in
              --includedir)
                printf '%s\n' "$merged_include"
                exit 0
                ;;
              --libdir)
                printf '%s\n' "$merged_lib"
                exit 0
                ;;
            esac
          done

          if printf '%s\n' "$@" | grep -qx -- '--cxxflags'; then
            "$real_llvm_config" "$@" | sed "s|$real_include|$merged_include|g"
            exit 0
          fi

          exec "$real_llvm_config" "$@"
        '';
        llvmInstall = pkgs.symlinkJoin {
          name = "llvm-mlir-21";
          paths = [
            llvmConfigWrapper
            llvmPkgs.tblgen
            llvmPkgs.llvm
            llvmPkgs.llvm.dev
            llvmPkgs.llvm.lib
            llvmPkgs.mlir
            llvmPkgs.mlir.dev
          ];
          # patch paths for MLIR's mlir-tblgen dependency in the installed CMake config
          postBuild = ''
            # The installed MLIRConfig.cmake sets MLIR_TABLEGEN_EXE to the bare
            # name "mlir-tblgen".  Ninja treats that as a file dependency relative
            # to each build sub-directory, which fails when using a pre-built LLVM.
            # Replace it with the absolute path so both command execution and
            # dependency tracking work.
            target="$out/lib/cmake/mlir/MLIRConfig.cmake"
            original=$(readlink -f "$target")
            rm "$target"
            sed 's|set(MLIR_TABLEGEN_EXE "mlir-tblgen")|set(MLIR_TABLEGEN_EXE "'"$out"'/bin/mlir-tblgen")|' \
              "$original" > "$target"
          '';
        };

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
            llvmPkgs.clang
            llvmPkgs.libclang
            llvmPkgs.tblgen
            llvmPkgs.llvm
            llvmPkgs.mlir
            llvmPkgs.llvm.dev
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
          CUDA_TOOLKIT_PATH = if isDarwin then "" else "${cudaToolkit}";
          CUDA_TILE_USE_LLVM_INSTALL_DIR = "${llvmInstall}";
          LLVM_CONFIG_PATH = "${llvmInstall}/bin/llvm-config";
          LLVM_DIR = "${llvmInstall}/lib/cmake/llvm";
          MLIR_DIR = "${llvmInstall}/lib/cmake/mlir";
          LLVM_INCLUDE_DIRS = "${llvmIncludeRoot}/include";
          LLVM_LIBRARY_DIR = "${llvmPkgs.llvm.lib}/lib";
          LIBCLANG_PATH = "${llvmPkgs.libclang.lib}/lib";
          MLIR_SYS_210_PREFIX = "${llvmInstall}";
          TABLEGEN_210_PREFIX = "${llvmInstall}";

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath ([
            pkgs.libffi
            pkgs.libxml2
            llvmPkgs.libclang.lib
            llvmPkgs.llvm.lib
            llvmPkgs.mlir
          ] ++ pkgs.lib.optionals (!isDarwin) [ cudaToolkit ]);

          shellHook = ''
            export PATH="${llvmInstall}/bin:$PATH"
            export CMAKE_PREFIX_PATH="${llvmInstall}:$CMAKE_PREFIX_PATH"
          '' + pkgs.lib.optionalString (!isDarwin) ''
            export PATH="${cudaToolkit}/bin:$PATH"
            # GPU driver libs: NixOS provides /run/opengl-driver/lib; on other
            # distros, symlink just the NVIDIA libs into a temp dir so we don't
            # pull in the host glibc.
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            else
              _nv_drv_dir=$(mktemp -d /tmp/nix-nvidia-driver.XXXXXX)
              for d in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/lib /usr/lib64; do
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

            if [ ! -d cuda-tile-rs/cuda-tile/.git ] && [ ! -f cuda-tile-rs/cuda-tile/CMakeLists.txt ]; then
              echo "Initializing cuda-tile submodule..."
              git submodule update --init --recursive
            fi

            echo ""
            echo "cutile-rs dev shell"
            echo " ${if isDarwin then "✗" else "✓"} CUDA  ${if isDarwin then "(not available on macOS)" else "$CUDA_TOOLKIT_PATH"}"
            echo " ✓ LLVM  $(llvm-config --version 2>/dev/null)"
            echo " ✓ Rust  $(rustc --version 2>/dev/null | awk '{print $2}')"
          '' + pkgs.lib.optionalString isDarwin ''
            echo ""
            echo "macOS: MLIR-only mode. Run: cargo run -p cutile-examples --example compile_only --no-default-features"
          '' + ''
            echo ""
          '';
        };
      }
    );
}
