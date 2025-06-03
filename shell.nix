{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python38; # using Python 3.8.20 from the exported environment
in
pkgs.mkShell {
  buildInputs = [ python pkgs.python38Packages.virtualenv ];

  shellHook = ''
    # Create and activate a virtualenv if not already present
    if [ ! -d .venv ]; then
      ${python.interpreter} -m venv .venv
    fi
    source .venv/bin/activate

    # Install pinned dependencies
    pip install --upgrade pip
    pip install -r requirements.txt
  '';
} 