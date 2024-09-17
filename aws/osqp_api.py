from benchmark_gen import problem_instance_gen
import warnings
warnings.filterwarnings("ignore")

from inst_set import Compiler
import src_helper 

import logging
from utils import data_pack_num
from utils import pad_problem_matrices

from utils import data_pack_num
from utils import pad_problem_matrices

import subprocess
import pandas as pd

class osqpFPGA:
	def __init__(self, 
			  hbm_pc,
			  fpga_id,
			  bitstream):
		self.hbm_pc = hbm_pc
		self.isca_c = data_pack_num * hbm_pc
		self.bitstream = bitstream
		self.fpga_id=fpga_id
		self.elf_on_fpga='./temp/test.fpga'

	def setup(self, P, q, A, l, u):
		qp_problem={'P':P, 'q':q, 'A':A, 'l': l, 'u':u}
		cu_dict = {}
		scalars={'sigma':1e-6,
				'rho':0.1 }
		scalars['isca_c'] = self.isca_c
		scalars['ori_dim_m'] = A.shape[0]	
		scalars['ori_dim_n'] = P.shape[1]	

		scalars['pdim_m'], scalars['pdim_n'], scalars['m_padding'], scalars['n_padding']=pad_problem_matrices(
			scalars['ori_dim_m'], 
			scalars['ori_dim_n'],
			self.isca_c)

		scalars['pdim_max']=max(scalars['pdim_m'], scalars['pdim_n'])

		helper_func=getattr(src_helper, 'osqp_indirect')
		helper_func(cu_dict, qp_problem, scalars)

		logging.debug("dim_n %d, %d dim_m %d, %d", 
					scalars['ori_dim_n'], 
					scalars['pdim_n'], 
					scalars['ori_dim_m'], 
					scalars['pdim_m']) 

		o_compiler = Compiler(self.hbm_pc, 
							cu_dict, 
							scalars['pdim_n'], 
							scalars['pdim_m'],
							scalars['n_padding'], 
							scalars['m_padding'] )

		o_compiler.program_gen('./aws/osqp_indirect.c')

		o_compiler.result_info(var_name='work_x', 
							vec_pack_len=o_compiler.unified_vec_pack_len)

		o_compiler.init_values()

		o_compiler.write_elf(self.elf_on_fpga)
		print('write elf to {}'.format(self.elf_on_fpga))

	def solve(self):
		""" TODO: Use pybind 11 to bind the host C program """
		program = './rsqp'
		args = ['-p', self.elf_on_fpga, 
		  '-d', str(self.fpga_id), 
		  '-x', self.bitstream]

		subprocess.run([program] + args, check=True)

		df= pd.read_csv('./temp/reg_layout.csv')
		content = pd.read_csv('./temp/reg_content.txt', header=None)
		df['Result']=content
		df.to_csv('./temp/reg_result.csv', index=False)
		reg_check_list =['max_iter', 'admm_iters', 'prim_res', 'dual_res',
						'rho_estimate', 'pcg_iters','norm_pcg_res', 'total_rho_update', 'total_pcg_iters']
		val_df = df[['Reg', 'Value', 'Result']]
		for item in reg_check_list:
			if item in df['Reg'].values:
				filtered_df = val_df[val_df['Reg'] == item]
				print("{}:  {}".format(item, 
							filtered_df['Result'].values[0]))

def main():
	logging.getLogger().setLevel(logging.DEBUG)

	qp_problem = problem_instance_gen(
		test_problem_name = 'Portfolio', 
		dim_idx = 0)

	prob = osqpFPGA(
		hbm_pc=4,
		fpga_id=0,
		bitstream='./temp/u50-4-u50-4000.xclbin')

	prob.setup(P = qp_problem['P'],
			q = qp_problem['q'],
			A = qp_problem['A'],
			l = qp_problem['l'],
			u = qp_problem['u'])

	prob.solve()

if __name__ == '__main__':
	main()