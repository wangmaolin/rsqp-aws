#!/bin/bash
set -e # Exit on the first error

echo "1 ====== Download aws-fpga ... =======" 
git clone https://github.com/aws/aws-fpga.git $AWS_FPGA_REPO_DIR

echo "2 ====== Install miniconda ... =======" 
sudo yum install wget
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

echo "3 ====== Install python libraries ... ====== " 
pip3 install -r requirements.txt

echo "4 ====== Compiling host program ... ====== "
make

echo "5 ====== Program FPGA ====== "
if [ `hostname` == "flatwhite" ]; then
	BITSTREAM=./temp/u280-1-ins1-4000.xclbin
	FPGA_ID=0000:c1:00.1
elif [ `hostname` == "labpc3" ]; then
	BITSTREAM=./temp/u50-1-ins1-4000.xclbin
	FPGA_ID=0000:01:00.1
else
	BITSTREAM=./temp/aws-1-aws-4000.awsxclbin
	FPGA_ID=0000:00:1d.0 
fi

xbutil program --device $FPGA_ID -u $BITSTREAM
