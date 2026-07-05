{ buildPythonPackage, fetchPypi, matplotlib, scipy, opencv4Full }:

buildPythonPackage rec {
  pname = "pyDHM";
  version = "1.0.5";
  format = "setuptools";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-iCPqla+W5ZHqeXE1vbICWErA7KbNZF8mt2N+1Yuft7g=";
  };

  propagatedBuildInputs = [
    matplotlib
    scipy
    opencv4Full
  ];

  doCheck = false; 
}