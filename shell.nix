{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.numpy
    python3Packages.opencv4
    python3Packages.matplotlib
    python3Packages.scipy
  ];

  shellHook = ''
    echo "Среда загружена. Доступны: python, numpy, opencv, matplotlib"
    echo "Для выполнения задания запустите: python task.py"
  '';
}
