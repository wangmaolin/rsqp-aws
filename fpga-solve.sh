#!/bin/bash

set -e # Exit on first error

for i in "$@"; do
  case $i in
    -hw=*)
      BIT_STREAM="${i#*=}"
      shift # past argument=value
      ;;
    -sw=*)
      ELF_NAME="${i#*=}"
      shift # past argument=value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

ELF_PATH=./temp
RELEASE_PATH=./temp

if [ ! -f $RELEASE_PATH/$BIT_STREAM ] || [ ! -f $ELF_PATH/$ELF_NAME ]; then
	echo $ELF_NAME "OR" $BIT_STREAM "DOES NOT EXIST"
	exit 0
fi

# Check the compatibility between sw & hw 
HW_HBM_PC=`echo $BIT_STREAM | cut -d '-' -f2`
HW_ARCH_CODE=`echo $BIT_STREAM | cut -d '-' -f3`
HW_CVB_HEIGHT=`echo $BIT_STREAM | cut -d '-' -f4 | sed 's/.xclbin//'`

SW_HBM_PC=`echo $ELF_NAME | cut -d '-' -f3`
SW_ARCH_CODE=`echo $ELF_NAME | cut -d '-' -f4`
SW_CVB_HEIGHT=`echo $ELF_NAME | cut -d '-' -f5 | sed 's/.fpga//'`
if [[ $HW_HBM_PC != $SW_HBM_PC ]]; then
	echo $HW_HBM_PC "DON't MATCH" $SW_HBM_PC 
	exit 0
fi

# if [ "$SW_CVB_HEIGHT" -gt "$HW_CVB_HEIGHT" ]; then
	# echo $SW_CVB_HEIGHT "EXCEED" $HW_CVB_HEIGHT 
	# exit 0
# fi

./rsqp -x $RELEASE_PATH/$BIT_STREAM -p $ELF_PATH/$ELF_NAME
