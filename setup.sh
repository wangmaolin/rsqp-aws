#!/bin/bash
set -e # Exit on the first error

# Install software

# Compile host program rsqp.cpp
make

# Program FPGA
HBM_PC=1
if [ `hostname` == "flatwhite" ]; then
	BITSTREAM=./temp/u280-1-ins1-4000.xclbin
	FPGA_ID=0000:c1:00.1
elif [ `hostname` == "labpc3" ]; then
	BITSTREAM=./temp/u50-1-ins1-4000.xclbin
	FPGA_ID=0000:01:00.1
else
	BITSTREAM=./temp/aws-$HBM_PC-aws-4000.awsxclbin
	FPGA_ID=0000:00:1d.0 
fi

xbutil program --device $FPGA_ID -u $BITSTREAM
