{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.matplotlib
    python3Packages.opencv4
  ];

  shellHook = ''
    if [ ! -d ".venv" ]; then
      python -m venv .venv
    fi

    source .venv/bin/activate

    if [ -f "dependencies.txt" ]; then
      pip install -r dependencies.txt
    fi
  '';
}
