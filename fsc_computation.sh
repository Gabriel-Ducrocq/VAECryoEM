#!/bin/bash
conda activate eman2

for i in {100..10000}
do
	e2proc3d.py dataset/heterogeneous_test/volumes/volume_$i.mrc dataset/heterogeneous_test/fsc_$i.txt --calcfsc=dataset/heterogeneous_test/output_cryoDRGN/all_volumes23/vol_$i.mrc  --apix=1.0
done