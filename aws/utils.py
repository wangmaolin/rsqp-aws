import re

def get_c_marco(macro_name, c_code):
	""" Extract macro defined integer from c file """
	with open(c_code, 'r') as file:
		c_code = file.read()
		pattern = r'#define\s+' + macro_name + r'\s+(\d+)'
		match = re.search(pattern, c_code)
		if match:
			integer_value = int(match.group(1))
		else:
			integer_value = None
		return integer_value

# data_pack_num = get_c_marco('DATA_PACK_NUM',
							# './unit00/src/constant_type.h')
data_pack_num = 16

def align_compute_padding(dim_n, isca_c):
	if dim_n % isca_c == 0:
		dim_n_pad_num = 0
	else:
		dim_n_pad_num = isca_c - dim_n % isca_c
	return dim_n + dim_n_pad_num, dim_n_pad_num

def pad_problem_matrices(ori_dim_m, ori_dim_n, isca_c):
	""" Preprocess the Problem Data"""
	pdim_m, m_padding = align_compute_padding(ori_dim_m, isca_c)
	pdim_n, n_padding = align_compute_padding(ori_dim_n, isca_c)
	return pdim_m, pdim_n, m_padding, n_padding 

def omega_rwc_bitwidth(stage_num):
	""" Compute the bitwidth of read, write and ctrl signal """
	ctrl_bitwidth = 2 * stage_num
	read_bitwidth = 15 - stage_num
	write_bitwidth = 15 - stage_num
	mul_sel_bitwidth = 1
	return ctrl_bitwidth, read_bitwidth, write_bitwidth, mul_sel_bitwidth

def rotate(list, n):
	return list[n:] + list[:n]