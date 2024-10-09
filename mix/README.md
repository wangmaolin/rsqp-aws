Mixed precision OSQP
===

## Motivation 

Using fp16 to replace fp32 benefits cloud applications like back-testing and edge applications like model predictive control (MPC).

## Research Question 1
Using reduce intermediate precision without affecting the solution accuray
 - Finance: profit of the portfolio
 - Control: precision

## Research Question 2
hardware benefit 
- Frequency increase 
- Area saved

## Requirements
python package: osqp, torch

To test different dtypes in ```osqp_pth.py```, requires a CUDA GPU
- torch.bfloat16 seems to work 
- torch.float16 and torch.float8_* doesn't work yet
