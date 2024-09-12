import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--content',type=str, required=True)
parser.add_argument('--vector',type=str, required=True)
parser.add_argument('--src-file',type=str, required=True)
parser.add_argument('--ground-truth',type=str, default=None)

def top_5_diff(a, b, diff):
	sorted_indices = np.argsort(diff)[::-1]
	for i in range(5):
		index = sorted_indices[i]
		print("{}\t{:.2e}\t{:.2e}\t{:.2e}".format(index, a[index], b[index], diff[index]))

def check_diff(a, b, var_name):
	if len(a)<len(b):
		a = np.pad(a, (0, len(b)-len(a)), mode='constant')
		
	# Calculate the maximum absolute difference
	abs_diff = np.abs(a - b)
	max_abs_diff = np.max(abs_diff)
	print("Maximum absolute difference: {:.2e}".format(max_abs_diff))
	top_5_diff(a, b, abs_diff)

	# Calculate the maximum relative difference
	rel_diff = np.abs(np.divide(a - b, a, out=np.zeros_like(a), where=a!=0))
	max_rel_diff = np.max(np.abs(rel_diff))
	print("Maximum relative difference: {:.2e}".format(max_rel_diff))
	top_5_diff(a, b, rel_diff)

	plt.subplot(2,2,1)
	plt.plot(a, label='truth')
	plt.plot(b, label='result')
	plt.legend()
	plt.title(var_name)

	plt.subplot(2,2,2)
	plt.plot(a-b, label='diff')
	plt.legend()

	sampleSize = 32
	plt.subplot(2,2,3)
	plt.plot(a[0:sampleSize], label='truth')
	plt.plot(b[0:sampleSize], label='result')
	plt.legend()

	random_sample = np.random.randint(0,len(a)-1, size=sampleSize)
	plt.subplot(2,2,4)
	plt.scatter(random_sample, a[random_sample], label='truth', 
			 marker='o',c='none', edgecolors='g')
	plt.scatter(random_sample, b[random_sample], label='result', 
			 marker='+', color='r')
	plt.legend()

	plt.savefig('./temp/result_compare.png')

def get_float_var(var_name, c_code):
	with open(c_code, 'r') as file:
		c_code = file.read()
		pattern = r'\bfloat\s+{}\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'.format(var_name)
		match = re.search(pattern, c_code)
		if match:
			initial_value = float(match.group(1))
			return initial_value
		else:
			return None

def get_assign_var(assign_var, c_code):
	with open(c_code, 'r') as file:
		c_code = file.read()
		pattern = r"\b" + re.escape(assign_var) + r"\b\s*=\s*([^;]+)"
		matches = re.findall(pattern, c_code)
		if matches:
			return matches[0]
		else:
			return None

def main():
	args = parser.parse_args()

	df= pd.read_csv('./temp/reg_layout.csv')
	content = pd.read_csv(args.content, header=None)
	df['Result']=content
	df.to_csv('./temp/reg_result.csv', index=False)
	reg_check_list =['max_iter', 'admm_iters', 'prim_res', 'dual_res',
					'rho_estimate', 'pcg_iters','norm_pcg_res', 'total_rho_update', 'total_pcg_iters','ut_scalar']
	val_df = df[['Reg', 'Value', 'Result']]
	for item in reg_check_list:
		if item in df['Reg'].values:
			filtered_df = val_df[val_df['Reg'] == item]
			print("{}:  {}".format(item, 
						  filtered_df['Result'].values[0]))

	""" Verify Vector Results """
	if args.ground_truth is not None and 'none' not in args.ground_truth:
		gt_vector = np.fromfile('./temp/'+args.ground_truth, dtype=np.float32)
		res_vector = np.loadtxt(args.vector)
		print('gt res len', len(gt_vector), len(res_vector))
		check_diff(gt_vector, res_vector[:len(gt_vector)], args.ground_truth)

if __name__ == '__main__':
	main()