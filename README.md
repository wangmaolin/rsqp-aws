Reconfigurable Solvers for QP running on AWS FPGAs
===
# Python Interface 
Example: `./aws/osqp_api.py`
```
prob = osqpFPGA()
prob.setup(P = qp_problem['P'],
		q = qp_problem['q'],
		A = qp_problem['A'],
		l = qp_problem['l'],
		u = qp_problem['u'])
prob.solve()
```

# C Interface 
see `./aws/osqp_api.c`

Example: `./aws/osqp_demo.c`

# Deployment 
We support on premise and AWS deployment with the following FPGA models:

| Instance Type | U50 | U280 | f1.2xlarge| f1.4xlarge| f1.16xlarge|
|:---:|---|---|---|---|---|
| Solver Cores | 16 | 32 | 7 | 14 | 56 |

## Example setup on AWS 
Launch an F1 instance with [FPGA Developer AMI](https://aws.amazon.com/marketplace/pp/prodview-gimv3gqbpe57k), 
then connect to the instance with the key pair 

`ssh -i "keyfile.pem" centos@ec2-hostIP.compute-1.amazonaws.com`

Clone this repo and run `./setup.sh` to 
- install the dependency 
- compile the host program
- program FPGA with bitstream

Setup environment variables
```
$ source $AWS_FPGA_REPO_DIR/vitis_runtime_setup.sh
$ source /opt/xilinx/xrt/setup.sh
```

Run the python interface example `python ./aws/osqp_api.py`

# Code Structure 
- Source program running on FPGA
	- OSQP indirect `./aws/osqp_indirect.c`
	- unit test `./aws/ut_spmv.c`, etc.
- Instruction set for different solver algorithms
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
	- `./aws/rsqp.cpp` download the compiled solver algorithm to FPGA

# Acknowledgement

This repo is based on the following research and open sourced projects:
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

- QP benchmark generation from osqp_benchmark
	- `./aws/benchmark_gen.py`, `./aws/control.py`, etc.
- pycparser
- osqp C API
