import sys
sys.path.append('../aws')

from benchmark_gen import problem_instance_gen
import warnings
warnings.filterwarnings("ignore")

import osqp
from osqp_pth import osqpTorch
	
def main():
	qp_problem = problem_instance_gen(
		test_problem_name = 'Portfolio', 
		dim_idx = 0)
	""" Research of using mixed precision"""
	prob_mp=osqpTorch()
	prob_mp.setup(P=qp_problem['P'], 
			q=qp_problem['q'],
			A=qp_problem['A'],
			l=qp_problem['l'],
			u=qp_problem['u'])
	prob_mp.solve()

	return
	""" Result of using standard precision"""
	prob=osqp.OSQP()
	prob.setup(scaling=0,
			P=qp_problem['P'], 
			q=qp_problem['q'],
			A=qp_problem['A'],
			l=qp_problem['l'],
			u=qp_problem['u'])

	official_results = prob.solve()
	print(official_results)

if __name__ == '__main__':
	main()
