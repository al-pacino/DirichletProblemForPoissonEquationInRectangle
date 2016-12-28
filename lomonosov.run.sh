#!/bin/bash

module add impi/5.0.1

sbatch -n $1 -p test -o "./results/time.$1.$2.txt" impi ./dirch $2 $2 $3

module rm impi/5.0.1
