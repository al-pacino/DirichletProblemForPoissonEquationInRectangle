#!/bin/bash

module add openmpi/1.8.4-gcc
module add impi/5.0.1

mpicxx -Wall -O2 -D"DIRCH_NO_OPENMP" -I./src src/main.cpp src/MpiSupport.cpp src/MathObjects.cpp src/MathFunctions.cpp -o dirch 2> errors.txt

module rm openmpi/1.8.4-gcc
module rm impi/5.0.1
