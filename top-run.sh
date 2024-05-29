#!/bin/sh
set -e # Exit on the first error

ARCH_CODE=sim
HBM_PC=1
CVB_SIZE=4000

BITSTREAM=aws-$HBM_PC-aws-4000.xclbin
xbutil program --device 0000:00:1d.0 -u ./temp/$BITSTREAM

# APP_NAME=SVM
APP_NAME=Control
# APP_NAME=Lasso
# APP_NAME=Huber
# APP_NAME=Portfolio

SCALE_IDX=0
# SCALE_IDX=8

# ALGO_SRC=./aws/osqp_indirect.c
# DEBUG_VAR=test_out
# DEBUG_VAR=none

ALGO_SRC=./aws/ut_spmv.c
# ALGO_SRC=./aws/ut_vecop.c
DEBUG_VAR=test_out

GROUND_TRUTH=$DEBUG_VAR

rm -f ./temp/*.fpga
rm -f ./temp/*.csv

python3 -u ./aws/toolchain.py\
	--output-dir ./temp\
	--arch-code $ARCH_CODE\
	--hbm-pc $HBM_PC\
	--app-name "$APP_NAME"\
	--scale-idx $SCALE_IDX\
	--src-file $ALGO_SRC\
  --result $DEBUG_VAR

# exit 0

ELF_PREFIX=`echo "$APP_NAME" | tr '[:upper:]' '[:lower:]'`-s$SCALE_IDX-$HBM_PC-$ARCH_CODE
SW_FILE=`ls temp/ | grep $ELF_PREFIX | head -n 1`

if [ -z "$SW_FILE" ]; then
    echo "NO ELF FILE FOUND!!!!!"
    exit 0
fi

./fpga-solve.sh -sw=$SW_FILE -hw=$BITSTREAM
SIM_RESULT_DIR="./temp"

python ./aws/reg_val_check.py\
 --content $SIM_RESULT_DIR/reg_content.txt\
 --vector $SIM_RESULT_DIR/result_vector.txt\
 --src-file $ALGO_SRC\
 --ground-truth $GROUND_TRUTH
