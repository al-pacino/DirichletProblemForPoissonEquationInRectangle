#!/bin/bash

rm ./dirch
mpicxx -Wall -O2 -D"DIRCH_NO_OPENMP" -I./src src/main.cpp src/MpiSupport.cpp src/MathObjects.cpp src/MathFunctions.cpp -o dirch 2> errors.txt

rm ./dirchomp
mpixlcxx_r -O2 -qsmp=omp -I./src src/main.cpp src/MpiSupport.cpp src/MathObjects.cpp src/MathFunctions.cpp -o dirchomp 2> errorsomp.txt
