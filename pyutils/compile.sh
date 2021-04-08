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

g++ -I$PYCONFIGDR/include \
    -I$PYCONFIGDR/include/python3.7m \
    -L$PYCONFIGDR/lib \
    model_ext.cpp \
    -Ofast -c -fpic -o model_ext.o 

g++ -I$PYCONFIGDR/include \
    -I$PYCONFIGDR/include/python3.7m \
    -L$PYCONFIGDR/lib \
    -shared -fpic \
    -o model_ext.so model_ext.o \
    -lboost_numpy37 -lboost_python37
