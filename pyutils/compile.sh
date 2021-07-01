# ====================================================================================== #
# Compilation file for firm model_ext.hpp as a shared library that can be imported into
# Python.
# 
# This should be modified for either fast or debugging installation.
# 
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
#!/bin/bash

PYCONFIGDR=/home/eddie/anaconda3/envs/corp
PY_VERSION=3.8

if [[ $1 = '-g' ]]; then
  echo 'compiling for debugging...must be switched off for code usage'
  g++ -I$PYCONFIGDR/include \
      -I$PYCONFIGDR/include/python$PY_VERSION \
      -L$PYCONFIGDR/lib \
      model_ext.cpp \
      -Og -c -fpic -o model_ext.o 
else
  g++ -I$PYCONFIGDR/include \
      -I$PYCONFIGDR/include/python$PY_VERSION \
      -L$PYCONFIGDR/lib \
      model_ext.cpp \
      -Ofast -c -fpic -o model_ext.o 
fi

g++ -I$PYCONFIGDR/include \
    -I$PYCONFIGDR/include/python$PY_VERSION \
    -L$PYCONFIGDR/lib \
    -shared -fpic \
    -o model_ext.so model_ext.o \
    -lboost_numpy${PY_VERSION/.} -lboost_python${PY_VERSION/.}
