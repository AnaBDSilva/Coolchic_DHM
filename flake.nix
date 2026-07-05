{
  description = "Hologram Compression";

  nixConfig = {
    extra-substituters = [
      "https://cache.nixos.org"
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/54170c54449ea4d6725efd30d719c5e505f1c10e";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};

    myPython = pkgs.python313.override {
      packageOverrides = self: super: {
        opencv4Full = super.opencv4Full.override {
          enableGtk3 = true;
        };

        pyDHM = self.callPackage ./nix/pyDHM.nix {
          opencv4Full = self.opencv4Full;
        };

        noteleDHM = self.callPackage ./nix/noteleDHM.nix {
          opencv4Full = self.opencv4Full;
        };
      };
    };

    fhsEnv = pkgs.buildFHSEnv {
      name = "cool-chic-env";

      targetPkgs = pkgs:
        with pkgs; [
          (myPython.withPackages (python-pkgs: [
            python-pkgs.pandas
            python-pkgs.numpy
            python-pkgs.matplotlib
            python-pkgs.pymatreader
            python-pkgs.opencv4Full
            python-pkgs.scipy
            python-pkgs.scikit-image
            python-pkgs.jupyter
            python-pkgs.notebook
            python-pkgs.ipykernel
            python-pkgs.pyDHM
            python-pkgs.noteleDHM
            python-pkgs.pip
            python-pkgs.pillow
            python-pkgs.setuptools
            python-pkgs.wheel

            # --- cool-chic deps ---
            python-pkgs.torch-bin
            python-pkgs.torchvision-bin
            python-pkgs.pybind11
            python-pkgs.numpy
            python-pkgs.imageio
            python-pkgs.scikit-image
            python-pkgs.psutil
          ]))

          # --- system tools ---
          glibc.bin
          gcc
          gnumake
          cmake
          ffmpeg
        ];

      extraBuildCommands = ''
        mkdir -p sbin
        ln -s ${pkgs.glibc.bin}/bin/ldconfig sbin/ldconfig
      '';

      profile = ''
                export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

                export TRITON_LIBCUDA_PATH="/run/opengl-driver/lib"
                export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
                export TORCH_INDUCTOR_MAX_AUTOTUNE=0

                export CC="gcc"
                export CXX="g++"

                export COOL_CHIC_DIR="$(pwd)/Cool-Chic"
                export PYTHONPATH="$COOL_CHIC_DIR:$(pwd):$PYTHONPATH"

                # activate local venv if it exists
                if [ -f "$(pwd)/.venv-coolchic/bin/activate" ]; then
                  source "$(pwd)/.venv-coolchic/bin/activate"

                  pip install ipykernel -q
                  python -m ipykernel install --user --name=venv-coolchic --display-name "Python (VENV Cool-Chic)"
                fi

                alias cc-encode="python \$COOL_CHIC_DIR/cc_encode.py"
                alias cc-decode="python \$COOL_CHIC_DIR/cc_decode.py"

                if [ ! -d "$COOL_CHIC_DIR" ]; then
                  echo ""
                  echo "  [cool-chic] Repo not found. Run:  ./setup-coolchic.sh"
                  echo ""
                fi
      '';
    };
  in {
    devShells.${system}.default = fhsEnv.env;
  };
}
