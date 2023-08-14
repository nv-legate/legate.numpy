#!/bin/bash

legate_root=`python -c 'import legate.install_info as i; from pathlib import Path; print(Path(i.libpath).parent.resolve())'`
echo "Using Legate at $legate_root"

`python -c 'import cunumeric.install_info as i; from pathlib import Path; print(f"export cunumeric_root={Path(i.libpath).parent.resolve()}")'`
echo "Using cunumeric at $cunumeric_root"

rm -rf build
cmake -B build -S . -D legate_core_ROOT="$legate_root" -D cunumeric_ROOT="$cunumeric_root" -D CMAKE_BUILD_TYPE=Debug
cmake --build build -j 8
