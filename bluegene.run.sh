#!/bin/bash

mpisubmit.bg -n $2 -w $1:00 -m smp \
	--stdout "./results$4/time.$2.$3.\$(jobid).txt" \
	./dirch$4 -- $3 $3
