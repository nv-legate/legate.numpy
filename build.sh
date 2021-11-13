#!/bin/bash
#touch src/matrix/matvecmul.cc
#touch src/fused/fused_op.cc
cd ../legate.core/
python setup.py --cuda --with-cuda /usr/local/cuda/ --arch pascal 
cd ../legate.numpy/
python setup.py --with-core /home/shiv1/truefusedGPU/legate.core/install/  --verbose
#../legate.core/install/bin/legate examples/testbench/test.py --cpus 2
