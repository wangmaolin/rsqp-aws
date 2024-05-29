Reconfigurable Solvers for QP running on AWS FPGAs
===

# Core components
- Source program running on FPGA
	- OSQP indirect `./aws/osqp_indirect.c`
	- unit test `./aws/ut_spmv.c`, etc.
- Instruction set for QP solver 
	- `./aws/inst_set.py`
	- example IR generated `./temp/ir_table.csv`
- compiler 
	- `./aws/toolchain.py`
	- `./aws/src_helper.py` prepare problem data P, q, A, l, u
	- `./aws/mib_sched.py` spatial and temporal instruction interleaving
- Micro Architecture Template 
	- Architecture scaling generation flow 
		- Support different architecture width and # of solver instances on a single FPGA  
		- not included in this repo yet
	- Example uArch implemented on AWS VU9P FPGA `./temp/aws-1-aws-4000.xclbin` 
		- too big to put on Github
- Source program running on CPU
	- `./aws/rsqp.cpp` download the compiled executable to FPGA

# Other open-sourced components
- QP benchmark generation from osqp_benchmark
	- `./aws/benchmark_gen.py`, `./aws/control.py`, etc.
- pycparser

# CVXPYgen integration plan
CVXPYgen can use 
- `./aws/rsqp.cpp` to call FPGA solver
- `./aws/src_helper.py` to compile the problem data P, q, A, l, u to run on the FPGA solver 
- interfaces for running batch QP solving with multiple solver instances on FPGA?
- we can keep optimizing the underlying uArch on FPGA and keep the same  interfaces for CVXPYgen

## AWS Setup and Instance Launch
Launch F1 instance(s) on AWS using this [Template](https://aws.amazon.com/marketplace/pp/prodview-zzeaoszfrkr7s)

| Instance Type | Maximum # of Parallel Solvers | USD per hour |
|:---:|---|---|
|f1.2xlarge | 7 | 1.65 |
|f1.4xlarge | 14 | 3.30 |
|f1.16xlarge | 56 | 13.20 |

The example launch flow 
- Subscribe to the AMI
- Continue to Configuration
- Continue to Launch
- Launch through EC2
- Create/Select Key pair

## Connect to the instance 
### Login with the key pair
`ssh -i "AC-CVXPY-RSA.pem" centos@ec2-3-236-136-102.compute-1.amazonaws.com`

### Check & Program FPGA
`source /opt/xilinx/xrt/setup.sh`

`xbutil examine`

xbutil program --device 0000:00:1d.0 -u ./temp/test.awsxclbin 

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

### Test script
```
source /opt/xilinx/xrt/setup.sh
./top-run.sh
```

This script will 
- install the dependency
- compile host program
- compile example SVM/LASSO problem data to the FPGA program
- run the FPGA solver and copy the solution back to `./temp/`

# Acknowledgement

This work is based on the following works:
```
@article{osqp,
  author  = {Stellato, B. and Banjac, G. and Goulart, P. and Bemporad, A. and Boyd, S.},
  title   = {{OSQP}: an operator splitting solver for quadratic programs},
  journal = {Mathematical Programming Computation},
  year    = {2020},
  volume  = {12},
  number  = {4},
  pages   = {637--672},
  doi     = {10.1007/s12532-020-00179-2},
  url     = {https://doi.org/10.1007/s12532-020-00179-2},
}

@inproceedings{wang2023rsqp,
  title={RSQP: Problem-specific Architectural Customization for Accelerated Convex Quadratic Optimization},
  author={Wang, Maolin and McInerney, Ian and Stellato, Bartolomeo and Boyd, Stephen and So, Hayden Kwok-Hay},
  booktitle={Proceedings of the 50th Annual International Symposium on Computer Architecture},
  pages={1--12},
  year={2023}
}

```