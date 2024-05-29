Reconfigurable Solvers for QP running on AWS FPGAs
===

## Introduction

## AWS Setup and Instance Launch
Launch F1 instance(s) on AWS using this [Template](https://aws.amazon.com/marketplace/pp/prodview-zzeaoszfrkr7s)

| Instance Type | Supported Parallel Solvers | USD per hour |
|:---:|---|---|
|f1.2xlarge | 3 | 1.65 |
|f1.4xlarge | 6 | 3.30 |
|f1.16xlarge | 24 | 13.20 |

Click these buttons on Webpages 
- Subscribe 
- Continue to Configuration
- Continue to Launch
- Launch through EC2
- Create/Select Key pair

## Connect to the instance 
### Login with the key pair
`ssh -i "AC-CVXPY-RSA.pem" centos@ec2-3-236-136-102.compute-1.amazonaws.com`

### Check the FPGA
`source /opt/xilinx/xrt/setup.sh`

`xbutil examine`

### Upload the Github repo && cd repo

### Setup miniconda for running the compiler 
```
sudo yum install wget
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

`pip3 install -r requirements.txt`

`make`

```
source /opt/xilinx/xrt/setup.sh
./top-run.sh
```