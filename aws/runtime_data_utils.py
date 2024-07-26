from scipy import sparse
from scipy.sparse import triu
import numpy as np
from benchmark_gen import problem_instance_gen
import warnings
warnings.filterwarnings("ignore")

def replace_np_inf(a, double_flag):
	if double_flag:
		a[a==np.inf]= 1e30
		a[a==-np.inf]= -1e30
		return a.astype(np.float64)
	else:
		a[a==np.inf]= 1e17
		a[a==-np.inf]= -1e17
		return a.astype(np.float32)

def convert_to_uint32(a):
	return a.astype(np.uint32)

def write_vec_to_file(vec, file_ptr, double_flag):
	if type(vec[0]) is np.float64:
		vec_to_write = replace_np_inf(vec, double_flag)
	elif type(vec[0]) is np.int32:
		vec_to_write = convert_to_uint32(vec)
	else:
		print("UNKNOWN VECTOR TYPE !!!!!!!!!!!!!!!")
		return
	vec_to_write.tofile(file_ptr)
	# print('vector first element: {}'.format(vec_to_write[0]))
	# print('vector last element: {}'.format(vec_to_write[-1]))

def generate_runtime_data(qp_problem, problem_name, double_flag):
	upper_P = triu(qp_problem['P'], format='csc')
	# Get problem dimension
	n = upper_P.shape[0]
	m = qp_problem['A'].shape[0]
	print('problem data n = {}, m = {}'.format(n, m))

	data_info = np.zeros(8, dtype=np.int32)
	data_info[0]=n
	data_info[1]=m
	""" A nzmax"""
	data_info[2]=len(qp_problem['A'].data)
	""" P nzmax"""
	data_info[3]=len(upper_P.data)
 
	if double_flag:
		problem_name += '_double'
	file_name ='./temp/'+problem_name+'.dat'

	with open(file_name, "wb") as problem_data_file: 
		""" Write data meta information"""
		data_info.tofile(problem_data_file)

		""" Write vectors """
		for vec_str in ['l', 'u', 'q']:
			print('----- writing vector '+vec_str)
			write_vec_to_file(qp_problem[vec_str], problem_data_file, double_flag)

		""" Write matrices """
		for mat_str in ['A']:
		# for mat_str in ['A', 'P']:
			print('----- writing matrix '+mat_str)
			write_vec_to_file(qp_problem[mat_str].data, problem_data_file, double_flag)
			write_vec_to_file(qp_problem[mat_str].indices, problem_data_file, double_flag)
			write_vec_to_file(qp_problem[mat_str].indptr, problem_data_file, double_flag)

		print('----- writing matrix P')
		write_vec_to_file(upper_P.data, problem_data_file, double_flag)
		write_vec_to_file(upper_P.indices, problem_data_file, double_flag)
		write_vec_to_file(upper_P.indptr, problem_data_file, double_flag)

def main():
	qp_problem = problem_instance_gen(
		test_problem_name = 'Control', 
		dim_idx = 0)
	generate_runtime_data(
		qp_problem=qp_problem,
		problem_name='demo_input', 
		double_flag=False)

if __name__ == '__main__':
	main()