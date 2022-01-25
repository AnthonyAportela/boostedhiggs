#!/bin/bash

python3 -m pip install correctionlib==2.0.0rc6

# make dir for output
mkdir outfiles

# run code
# pip install --user onnxruntime
python SCRIPTNAME --year YEAR --starti STARTNUM --endi ENDNUM --samples SAMPLE --processor PROCESSOR

#move output to eos
xrdcp -f outfiles/* EOSOUTPKL


