from benchmark_gen import problem_instance_gen
import warnings
warnings.filterwarnings("ignore")

from inst_set import Compiler
import src_helper 

import logging
import os
from utils import data_pack_num
from utils import pad_problem_matrices

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--arch-code',type=str, required=True)
parser.add_argument('--hbm-pc', type=int, required=True)
parser.add_argument('--output-dir',type=str,  required=True)
parser.add_argument('--app-name',type=str, default='Lasso')
parser.add_argument('--scale-idx', type=int, default=0)
parser.add_argument('--src-file',type=str, required=True)
parser.add_argument('--result',type=str, default=None)

def main():
	logging.getLogger().setLevel(logging.DEBUG)
	args = parser.parse_args()
	isca_c = data_pack_num * args.hbm_pc
	cu_dict = {}

	""" Future Release Inteface: qp_problem P, q, A, l, u """
	qp_problem = problem_instance_gen(
		test_problem_name = args.app_name, 
		dim_idx = args.scale_idx)

	scalars={'sigma':1e-6,
			'rho':0.1 }
	scalars['isca_c'] = isca_c
	scalars['ori_dim_m'] = qp_problem['A'].shape[0]	
	scalars['ori_dim_n'] = qp_problem['P'].shape[1]	

	scalars['pdim_m'], scalars['pdim_n'], scalars['m_padding'], scalars['n_padding']=pad_problem_matrices(
		scalars['ori_dim_m'], 
		scalars['ori_dim_n'],
		isca_c)
	scalars['pdim_max']=max(scalars['pdim_m'], scalars['pdim_n'])
	func_name=os.path.basename(args.src_file).split('.')[0]
	helper_func=getattr(src_helper, func_name)
	helper_func(cu_dict, qp_problem, scalars)

	logging.debug("func %s dim_n %d, %d dim_m %d, %d", 
				func_name,
				scalars['ori_dim_n'], 
				scalars['pdim_n'], 
				scalars['ori_dim_m'], 
				scalars['pdim_m']) 

	o_compiler = Compiler(args.hbm_pc, 
					   	cu_dict, 
					   	scalars['pdim_n'], 
					   	scalars['pdim_m'],
						scalars['n_padding'], 
						scalars['m_padding'] )

	o_compiler.program_gen(args.src_file)

	if args.result is not None and 'none' not in args.result:
		o_compiler.result_info(var_name=args.result, 
						   vec_pack_len=o_compiler.unified_vec_pack_len)

	o_compiler.init_values()

	file_name = args.output_dir+'/test.fpga'
	o_compiler.write_elf(file_name)
	print('write elf to {}'.format(file_name))

if __name__ == '__main__':
	main()