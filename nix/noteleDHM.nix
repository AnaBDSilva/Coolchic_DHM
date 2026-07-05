{ lib, buildPythonPackage, fetchFromGitHub
, numpy, matplotlib, opencv4Full, imageio, scikit-image, python}:

buildPythonPackage rec {
  pname = "noteleDHM";
  version = "ae40c6c8eebce200c975cf84c38b6577241ea9d0"; 

  src = fetchFromGitHub {
    owner = "OIRL";
    repo = "noteleDHM-Tool";
    rev = "main"; 
    sha256 = "sha256-wGCJ0EsC9M8UB79a5rDPnafVl/JyB1DnzR+eGHy9R9I=";
  };

  propagatedBuildInputs = [
    numpy
    matplotlib
    opencv4Full
    imageio
    scikit-image
  ];

  format = "other"; 

installPhase = ''
    TARGET_DIR=$out/lib/python${python.pythonVersion}/site-packages/noteleDHM
    mkdir -p $TARGET_DIR
    
    cp -r * $TARGET_DIR/
    
    touch $TARGET_DIR/__init__.py
  '';

  doCheck = false;
}