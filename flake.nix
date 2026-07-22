{
  description = "Hologram Compression";

  nixConfig = {
    extra-substituters = [
      "https://cache.nixos.org"
    ];
    extra-trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
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
      name = "hologram-env";

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
        export CC="gcc"
        export CXX="g++"
      '';
    };
  in {
    devShells.${system}.default = fhsEnv.env;
  };
}