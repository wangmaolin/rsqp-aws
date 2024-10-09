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

## Methods
Experiment dtype in ```osqp_pth.py```
