import numpy as np
import pandas as pd
import logging

from pycparser import c_ast
from pycparser import parse_file

from utils import data_pack_num
from utils import get_c_marco
from utils import omega_rwc_bitwidth

def get_Decl_info(node):
	if node.init is not None:
		init_flag = 'yes'
		if hasattr(node.init, 'value'):
			init_value = float(node.init.value)
		else:
			""" init negative values """
			init_value = -float(node.init.expr.value)
	else:
		init_flag = 'unset'
		init_value= 0.0

	id_type_record =[node.name, node.type.type.names[0], node.coord, init_flag, init_value]
	return id_type_record
def lookup_var_type(df, var):
	assert hasattr(var, 'name')
	id=var.name
	id_record = df.loc[df['id'] == id]
	""" the ID should be unique """
	assert len(id_record) == 1, logging.debug('id not unique', id)
	id_type = id_record.iloc[0]['type']
	return id_type
def df_insert_row(df, row):
	df.loc[len(df)] = row
def cls_name(node):
	return node.__class__.__name__

class Compiler(c_ast.NodeVisitor):
	def __init__(self,
				 hbm_pc,
				 cu_dict,
				 pdim_n,
				 pdim_m,
				 n_padding,
				 m_padding,
				 max_data_size=pow(10,8)):
				 
		self.cu_dict = cu_dict
		self.hbm_pc = hbm_pc
		self.isca_c = hbm_pc * data_pack_num
		""" For generate omega instructions """
		self.pipeStage = np.log2(self.isca_c).astype(int)
		ctrlBW, readBW, writeBW = omega_rwc_bitwidth(self.pipeStage)
		assert readBW == writeBW
		self.readBitShift = ctrlBW
		self.writeBitShift = ctrlBW + readBW

		""" Packed problem dimension """
		self.sol_vec_pack_len = pdim_n//self.isca_c
		self.con_vec_pack_len = pdim_m//self.isca_c
		self.unified_vec_pack_len = max(self.con_vec_pack_len, self.sol_vec_pack_len)
		self.sol_cut_off_bank = self.isca_c-n_padding
		self.con_cut_off_bank = self.isca_c-m_padding
		""" use the other end on CVB as register file,
		 	omega instruction grows from the start, 
			check rf_read and rf_write in top_unit.cpp,
			make sure this equals to TETRIS_HEIGHT-1  """
		self.register_file_offset = 3999

		""" Instruction Types """
		self.cu_inst_num = 0
		self.inst_halt = 0
		self.inst_branch = 1
		self.inst_nop = 2
		self.inst_cvb_write = 3
		self.inst_axpby= 4
		self.inst_load_cvb = 5
		self.inst_dot = 6
		self.inst_rf_write = 6
		self.inst_scalar_op = 7
		self.inst_norm_inf = 8
		""" The unified instruction to rule it all"""
		self.inst_omega = 9
		""" broadcast one register to all banks """
		self.inst_cvb_broadcast = 10
		""" Memory Regions """
		self.uint32_pack_num = data_pack_num
		self.indice_mem_words = 0
		self.uint32_data_region = np.zeros(max_data_size, dtype=np.uint32)
		self.uint32_data_pointer = 0
		self.stage_size_recorder = []

		self.nnz_mem_words = 0
		self.nnz_data_region = []
		for _ in range(self.hbm_pc):
			self.nnz_data_region.append(np.zeros(max_data_size, dtype=np.float32))
		self.nnz_data_region_pointer = 0

		self.rhs_mem_words = 0
		self.rhs_data_region = []
		for _ in range(self.hbm_pc):
			self.rhs_data_region.append(np.zeros(max_data_size, dtype=np.float32))
		self.rhs_data_region_pointer = 0

		self.mem_ground_truth_loc = 0
		self.mem_verify_loc = 0
		self.verify_vec_pack_len = 0

		self.program_info = np.zeros((4, 4), dtype=np.uint32)
		self.info_ptr = 0

		self.RegFileSize = 64
		self.register_file = np.zeros(self.RegFileSize, dtype=np.float32)

		"""
		program info are 128 bits aligned
			- [0]: file header magic number: 2135247942
			- [1]: inst_rom_pack_size
			- [2]:
			- [3]:

			- [4]:
			- [5]: stage_5_pack_size or indice mem pack words
			- [6]:
			- [7]: mem data words

			- [8]: mem ground truth loc
			- [9]: mem verify loc
			- [10]: verify vec pack len
			- [11]: mem sol words

			- [12]: HBM PC NUM
			- [13]: 0
			- [14]: nnz_mem_words
			- [15]: rhs_mem_words assert equal """

		""" __init__ from previous EmitVisitor """
		self.symbol_table = pd.DataFrame(columns=['id','type','src', 'init_flag', 'const_value'])

		""" use the tuple of operand and id type as the key to rule dict """
		self.binop_prod_rule={
			('+', 'vectorf', 'vectorf') : ('axpby','linear'),
			('-', 'vectorf', 'vectorf') : ('axpby','linear'),
			('<', 'vectorf', 'vectorf') : ('axpby','select_min'),
			('>', 'vectorf', 'vectorf') : ('axpby','select_max'),
			('*', 'vectorf', 'vectorf') : ('dot', None),

			('*', 'float', 'vectorf') : ('axpby_frac', None),

			('+', 'float', 'float') : ('scalar_op', 'add'),
			('-', 'float', 'float') : ('scalar_op', 'sub'),
			('*', 'float', 'float') : ('scalar_op', 'mul'),
			('/', 'float', 'float') : ('scalar_op', 'div'),
			('%', 'float', 'float') : ('scalar_op', 'mod'),

			('>', 'float', 'float') : ('scalar_op', 'GT'),
			('<', 'float', 'float') : ('scalar_op', 'LT'),
			('||', 'float', 'float') : ('scalar_op', 'OR'),
			('&&', 'float', 'float') : ('scalar_op', 'AND'),
		}

		self.inst_out_type={
			'axpby': 'vectorf',
			'dot':'float',
			'axpby_frac': 'vectorf',
			'scalar_op':'float',
		}

		self.omegaconf_list = ['placeholder_cu_arg_info']

		self.axpby_type_dict ={'linear':0, 
						 'ew_prod':1, 
						 'ew_reciprocal':2, 
						 'select_min':3, 
						 'select_max': 4,
						 'set_scalar_conditional':5}  
		self.scalar_op_type_dict ={'add':1, 
								   'sub':2, 
								   'mul': 3,
								   'div': 4,
								   'select_max': 5,
								   'c_sqrt': 6,
								   'select_min': 7,
								   'mod':8,
								   'GT':9,
								   'LT':10,
								   'AND':11,
								   'OR':12,
								   }  

		""" binary op temp var counter """ 
		self.temp_var_idx=0
		self.const_var_idx=0

		""" buffer to hold fractions of axpby""" 
		self.axpby_buffer={'s_a': None,
							'v_x':None,
							's_b':None,
							'v_y':None,
							'frac_num':0}

		""" IR table"""
		self.ir_table = pd.DataFrame(columns=['inst_type', 'op_type','result',
										'v0','v1','s0','s1','length'])

		""" Two way partition for vector input of axpby instructions """
		self.bp_graph_nodes = []

		""" vector memory layout """
		self.rhs_hbm = ['register_file']
		self.lr_layout = pd.DataFrame(columns=['LHS', 'RHS'])

		""" register file layout with pre-defined map """
		self.reg_onchip = []
		self.reg_layout = pd.DataFrame(columns=['Reg', 'Value'])

		""" __init__ from Linker """
		# info_pack_size = get_c_marco('INFO_PACK_SIZE','./unit00/src/constant_type.h')
		info_pack_size = 64
		self.cu_arg_info = np.zeros(data_pack_num*info_pack_size, dtype=np.uint32)

		self.cvb_offset_dict = {
			'sol': self.sol_vec_pack_len,
			'con': self.con_vec_pack_len,
			'any': self.unified_vec_pack_len
			}

		self.cut_off_bank_dict = {
			'sol': self.sol_cut_off_bank,
			'con': self.con_cut_off_bank,
			'any': 0 
			}

	def IdxToBin(self, idx):
		""" Turn the binary representation of an int32 number to a byte array """
		cvt_temp = np.array([idx]).view(np.uint8)
		return np.unpackbits(cvt_temp, bitorder = 'little')[:self.pipeStage]

	def offset_translate(self, offset_str):
		offset_arr = offset_str.split('_')
		assert len(offset_arr) % 2 == 0
		total_offset = 0
		for item in range(0, len(offset_arr), 2):
			o_unit = self.cvb_offset_dict.get(offset_arr[item])
			assert o_unit is not None
			o_cnt = int(offset_arr[item+1])
			total_offset += o_cnt * o_unit
		return total_offset

	def cut_off_translate(self, offset_str):
		offset_arr = offset_str.split('_')
		assert len(offset_arr) == 2
		return self.cut_off_bank_dict.get(offset_arr[0])

	def dynamic_config(self, config_name):
		offset_item = self.lookup_omegaconf(config_name)
		self.cu_arg_info[data_pack_num*offset_item + 0] = self.nnz_mem_offset +\
			  (self.nnz_data_region_pointer//data_pack_num)
		self.cu_arg_info[data_pack_num*offset_item + 2] = len(self.cu_dict[config_name])//self.isca_c
		self.add_vector_nnz_mem(self.cu_dict[config_name])

	def compute_vecbuf_base_loc(self, vecbuf_addr):
		return vecbuf_addr * self.unified_vec_pack_len

	def write_elf(self, file_name):
		""" make sure the added mem indice contents are aligned """
		assert self.cu_inst_num % 4 == 0

		for item in self.stage_size_recorder:
			assert(item % data_pack_num == 0)

		inst_rom_pack_size = self.stage_size_recorder[0]//data_pack_num
		stage_2_pack_size = self.stage_size_recorder[1]//data_pack_num

		""" set program info """
		self.add_info([2135247942,
					   inst_rom_pack_size,
					   0,
					   0])

		self.add_info([0,
					   stage_2_pack_size,
					   0,
					   0])

		self.add_info([self.mem_ground_truth_loc,
					   self.mem_verify_loc,
					   self.verify_vec_pack_len,
					   0])

		self.add_info([self.hbm_pc,
					   0,
					   0,
					   self.nnz_mem_words+self.rhs_mem_words])

		with open(file_name, "wb") as f:
			""" Write program meta information"""
			self.program_info.tofile(f)

			""" Write to mem_indice """
			self.uint32_data_region[0:self.uint32_data_pointer].tofile(f)

			""" Write to mem_nnz"""
			for i in range(self.hbm_pc):
				self.rhs_data_region[i][0:self.rhs_data_region_pointer].tofile(f)
				self.nnz_data_region[i][0:self.nnz_data_region_pointer].tofile(f)

	def add_axpby_inst(self,
					   alpha_addr,
					   beta_addr,
					   src_tetris_addr,
					   src_vf_addr,
					   dst_addr,
					   op_type,
					   gamma_addr,
					   pack_size,
					   program_end=False):
		""" generate the instruction for ax plus by instruction """
		op0 = self.inst_axpby
		op1 = alpha_addr +\
			(beta_addr<<6) +\
			(src_tetris_addr<<12) +\
			(src_vf_addr<<18) +\
			(dst_addr<<24)
		op2 = op_type +\
			(gamma_addr<<6) 
		op3 = pack_size
		self.add_inst([op0, op1, op2, op3], program_end)

	def add_load_cvb_inst(self,
					   src_addr,
					   src_sel_lhs,
					   cvb_offset,
					   pack_size,
					   program_end=False):
		""" generate the instruction for ax plus by instruction """
		op0 = self.inst_load_cvb
		op1 = src_addr+\
			(src_sel_lhs<<6)
		op2 = cvb_offset
		op3 = pack_size
		self.add_inst([op0, op1, op2, op3], program_end)

	def add_cvb_write_inst(self,
					   dst_addr,
					   dst_sel_lhs,
					   cvb_offset,
					   pack_size,
					   cut_off_bank,
					   program_end=False):
		""" generate the instruction for ax plus by instruction """
		op0 = self.inst_cvb_write
		op1 = dst_addr+\
			(dst_sel_lhs<<6)+\
			(cut_off_bank<<16)
		op2 = cvb_offset
		op3 = pack_size
		self.add_inst([op0, op1, op2, op3], program_end)

	def add_norm_inf_inst(self,
					   src_addr,
					   dst_addr,
					   src_sel_lhs,
					   pack_size,
					   program_end=False):
		""" generate the instruction for ax plus by instruction """
		op0 = self.inst_norm_inf
		op1 = src_addr+\
			(dst_addr<<6)+\
			(src_sel_lhs<<12)
		op2 = pack_size
		self.add_inst([op0, op1, op2, 0], program_end)
	def add_dot_inst(self,
					 sel_norm,
					 src_tetris_addr,
					 src_vf_addr,
					 dst_reg,
					 pack_size,
					 program_end=False):
		""" generate the instruction for dot instruction """
		op0 = self.inst_dot
		op1 = src_tetris_addr +\
			(src_vf_addr<<6) +\
			(dst_reg<<12)+\
			(sel_norm<<18)
		op2 = pack_size
		self.add_inst([op0, op1, op2, 0], program_end)

	def add_rf_write_inst(self,
					dst_rf_addr,
					loc_addr,
					bank_addr,
					program_end=False):
		""" generate the instruction for dot instruction """
		op0 = self.inst_rf_write
		op1 = dst_rf_addr
		op2 = loc_addr
		op3 = bank_addr
		self.add_inst([op0, op1, op2, op3], program_end)
		return 

		""" Replace with the omega inst:
			if remove the extra rf_write instruction 6 and 10
			hbm4-cvb4000 u50 frequency increases 177Mhz -> 182Mhz
			hbm2-cvb4000 u50 frequency increases 243Mhz -> 216Mhz? wierd 
			problem write address bit width not enough,
			solution 1 seperate read and write offset, 
				use more fields in  
			solution 2 change the rf to the begining 
				and adjust all instruction cvb_offset accrodingly 
					- load cvb
					- cvb write
					- omega instruction
			TODO: solution 2 is better, still use 6 bits RF address but 
			leave 8 bits/256 register space

			TODO: add mechanisim to warn read and write field overflow 
			limited network program
			"""
		hbmInst = np.zeros(self.isca_c, dtype=np.uint32)
		""" set default write to null """
		nullAddr = self.register_file_offset
		hbmInst += (nullAddr<<self.writeBitShift)
		srcRF = bank_addr
		srcLoc = loc_addr
		dstRF = dst_rf_addr % self.isca_c
		dstLoc = self.register_file_offset-1-(dst_rf_addr>>self.pipeStage)
		print('srcRF', 'srcLoc', 'dstRF', 'dstLoc')
		print(srcRF, srcLoc, dstRF, dstLoc)

		hbmInst[srcRF] += (srcLoc<<self.readBitShift) 
		""" unset default write first"""
		hbmInst[dstRF] -= (nullAddr<<self.writeBitShift)
		hbmInst[dstRF] += (dstLoc<<self.writeBitShift) 
		dstBin = self.IdxToBin(dstRF)
		srcBin = self.IdxToBin(srcRF)
		flipFlags = np.bitwise_xor(srcBin, dstBin)
		routeCtrl = np.zeros(self.isca_c, dtype=np.uint32) 
		workNode = srcRF
		for stage_item in range(self.pipeStage):
			nodeSignal = np.zeros(self.isca_c, dtype=np.uint32) 
			flipBit = flipFlags[stage_item]
			workNode = workNode ^ (flipBit<<stage_item)
			if flipBit:
				nodeSignal[workNode] = 2
			else:
				nodeSignal[workNode] = 1
			routeCtrl += (nodeSignal<<(2*stage_item))
			# print(workNode, nodeSignal)
		hbmInst += routeCtrl

		inst_name='rf_write_'+str(len(self.omegaconf_list))
		self.omegaconf_list.append(inst_name)

		hbm_addr=self.nnz_mem_offset+\
			  (self.nnz_data_region_pointer//data_pack_num)
		""" put the rf write omega instruction to HBM """
		self.add_vector_nnz_mem(hbmInst.view(np.float32))
		self.add_omega_inst(hbm_addr=hbm_addr,
						 	hbm_pack_size=1,
							l2_opcode=2, 
							cvb_offset=0)

	def add_cvb_broadcast_inst(self,
					src_rf_addr,
					loc_addr,
					program_end=False):
		""" generate the instruction for dot instruction """
		op0 = self.inst_cvb_broadcast
		op1 = src_rf_addr
		op2 = loc_addr
		self.add_inst([op0, op1, op2, 0], program_end)
		""" TODO replace with omega inst """

	def add_scalar_op_inst(self,
						   src_0_reg,
						   src_1_reg,
						   scalar_op,
						   dst_reg,
						   imme_flag=0,
						   program_end=False):
		""" generate the instruction for dot instruction """
		op0 = self.inst_scalar_op
		op1 = src_0_reg+\
			(src_1_reg<<6) +\
			(dst_reg<<12) +\
			(scalar_op<<18) +\
			(imme_flag<<22)
		self.add_inst([op0, op1, 0, 0], program_end)
	def add_branch_inst(self,
						src_0_reg,
						force_jump,
						jump_address,
						program_end=False):
		""" generate the instruction for dot instruction """
		op0 = self.inst_branch
		self.add_inst([op0, src_0_reg, force_jump, jump_address], program_end)

	def add_omega_inst(self, 
						hbm_addr,
						hbm_pack_size,	
						l2_opcode,
					  cvb_offset,
					  input_mode=0,
					  operator_mode=0,
					  program_end=False):
		op0 = self.inst_omega
		op1 = hbm_addr
		op2 = hbm_pack_size
		op3 = cvb_offset +\
			(l2_opcode<<16)+\
			(operator_mode<<24)+\
			(input_mode<<28)
		self.add_inst([op0, op1, op2, op3], program_end)

	def add_inst(self, op_list, program_end = False):
		self.add_vector_uint32(op_list, program_end)
		self.cu_inst_num +=1
	def add_vector_uint32(self, vec, record_stage_size = False):
		size = len(vec)
		assert size % 4 ==0
		self.uint32_data_region[self.uint32_data_pointer: self.uint32_data_pointer+ size]=vec
		self.uint32_data_pointer += size
		self.indice_mem_words += size
		if record_stage_size:
			self.stage_size_recorder.append(self.uint32_data_pointer)

	def add_info(self,  info_list):
		for idx, info in enumerate(info_list):
			self.program_info[self.info_ptr][idx]=info
		self.info_ptr += 1

	def result_info(self, var_name, vec_pack_len):
		vf_addr = self.vec_addr(var_name)
		self.mem_verify_loc = ((vf_addr * vec_pack_len)<<1)
		self.verify_vec_pack_len = vec_pack_len 

	def lhs_to_cvb(self, vec_name):
		if vec_name is None:
			self.add_inst([self.inst_nop, 0, 0, 0])
		else:
			assert vec_name in self.rhs_hbm, print(vec_name)
			src_vec_addr = self.vec_addr(vec_name)
			self.add_load_cvb_inst(src_addr=src_vec_addr,
								src_sel_lhs=0,
								cvb_offset=0,
							pack_size=self.unified_vec_pack_len)

	def result_to_rhs(self, vec_name):
		assert vec_name in self.rhs_hbm, print(vec_name)
		dst_vec_addr = self.vec_addr(vec_name)
		self.add_cvb_write_inst(dst_addr=dst_vec_addr,
							dst_sel_lhs=0,
							cvb_offset=0,
							pack_size=self.unified_vec_pack_len,
							cut_off_bank=self.isca_c
							)

	def add_vector_nnz_mem(self, vec):
		""" split the matrix nnz into different HBM PCs"""
		size = len(vec)
		""" merged col and nnz HBM so x2 align"""
		# assert size % (self.isca_c*2) == 0
		assert size % self.isca_c == 0
		vec_stride = vec.reshape(-1, data_pack_num)
		size_each_pc = size//self.hbm_pc
		region_start = self.nnz_data_region_pointer
		self.nnz_data_region_pointer += size_each_pc
		region_end = self.nnz_data_region_pointer
		for i in range(self.hbm_pc):
			self.nnz_data_region[i][region_start:region_end] = np.concatenate(vec_stride[i::self.hbm_pc, :])

		self.nnz_mem_words += size_each_pc

	def add_vector_rhs_mem(self, vec):
		""" split the rhs vector into different HBM PCs"""
		vec=self.unified_vector_container(vec)

		size = len(vec)
		assert size % self.isca_c == 0
		vec_stride = vec.reshape(-1, data_pack_num)
		size_each_pc = size//self.hbm_pc
		region_start = self.rhs_data_region_pointer
		self.rhs_data_region_pointer += size_each_pc
		region_end = self.rhs_data_region_pointer
		for i in range(self.hbm_pc):
			self.rhs_data_region[i][region_start:region_end] = np.concatenate(vec_stride[i::self.hbm_pc, :])

		self.rhs_mem_words += size_each_pc
	def unified_vector_container(self, vec):
		container_size = self.unified_vec_pack_len * self.isca_c
		assert container_size > 0 and container_size >= len(vec)
		vector_container = np.zeros(container_size)
		vector_container[:len(vec)]=vec
		return vector_container

	def insert_ir_table(self, inst_type, 
					 op_type=None, result=None, 
					 v0=None, v1=None, s0=None, s1=None, length=1):
		df_insert_row(self.ir_table, [inst_type, op_type, result, v0, v1, s0, s1, length])
	def add_axpby_frac(self, scalar, vector):
		assert self.axpby_buffer['frac_num'] < 2, \
		"buffer full: {}, {}".format(self.axpby_buffer['v_x'],
					   self.axpby_buffer['v_y'])

		if self.axpby_buffer['frac_num'] == 0:
			self.axpby_buffer['s_a']=scalar
			self.axpby_buffer['v_x']=vector
		elif self.axpby_buffer['frac_num'] == 1:
			self.axpby_buffer['s_b']=scalar
			self.axpby_buffer['v_y']=vector

		self.axpby_buffer['frac_num'] += 1
	def emit_axpby_buffer(self, result_name, op_type='linear'):
		assert self.axpby_buffer['frac_num'] < 3 and\
			self.axpby_buffer['frac_num'] > 0

		self.insert_ir_table(inst_type='nop') 
		self.insert_ir_table(inst_type='axpby', 
					   op_type=op_type,
					   result=result_name,
					   v0=self.axpby_buffer['v_x'],
					   s0=self.axpby_buffer['s_a'],
					   v1=self.axpby_buffer['v_y'],
					   s1=self.axpby_buffer['s_b'],
					   length=1)

		assert self.axpby_buffer['v_x'] is not None

		if op_type == 'linear':
			""" insert nop placeholder for new linear impl by omega """
			self.insert_ir_table(inst_type='nop') 
			if self.axpby_buffer['v_y'] is not None:
				self.insert_ir_table(inst_type='nop') 
				self.insert_ir_table(inst_type='nop') 
				self.insert_ir_table(inst_type='nop') 

		""" clear the buffer after emitting """ 
		self.axpby_buffer['s_a']=None
		self.axpby_buffer['v_x']=None
		self.axpby_buffer['s_b']=None
		self.axpby_buffer['v_y']=None
		self.axpby_buffer['frac_num']=0

	def temp_var_info(self, var_type, node):
		if not hasattr(node, 'name'):
			temp_name ='temp-'+str(self.temp_var_idx)
			self.temp_var_idx += 1
			node.name=temp_name
			df_insert_row(self.symbol_table, [node.name, var_type, node.coord, 'unset', 0.0])
		else:
			""" the result of the binop has been declared, 
			check if the declared type and the result type are the same """ 
			decl_type = lookup_var_type(self.symbol_table, node)
			assert decl_type == var_type
	def const_var_info(self, var_type, node):
		""" check if the constant exists """
		df = self.symbol_table
		float_value = float(node.value)
		id_record = df.loc[df['const_value'] == float_value]
		if len(id_record) >= 1:
			""" Note: the src loc of the constant with the same value
				is not recorded """
			node.name = df.at[id_record.index[0], 'id']
		else:
			const_name ='const-'+str(self.const_var_idx)
			self.const_var_idx += 1
			node.name=const_name
			df_insert_row(self.symbol_table, [node.name, var_type, node.coord, 'const', float_value])
	def const_fill_info(self, fill_value):
		""" check if the constant exists """
		df = self.symbol_table
		id_record = df.loc[df['const_value'] == fill_value]
		if len(id_record) >= 1:
			""" Note: the src loc of the constant with the same value
				is not recorded """
			const_name = df.at[id_record.index[0], 'id']
		else:
			const_name ='const-'+str(self.const_var_idx)
			self.const_var_idx += 1
			df_insert_row(self.symbol_table, [const_name, 'float', 'filled', 'const', fill_value])

		return const_name
			
	def visit_Decl(self, node):
		""" gather ID, type info from decl stmt"""
		df_insert_row(self.symbol_table, get_Decl_info(node))
	def visit_Constant(self, node):
		self.const_var_info('float', node)
	def visit_BinaryOp(self, node):
		for c in node:
			self.visit(c)
		""" choose machine idiom based on l-type and r-type """
		left_type = lookup_var_type(self.symbol_table, node.left)
		right_type = lookup_var_type(self.symbol_table, node.right)
		prod_head = (node.op, left_type, right_type)
		emit_inst, op_type = self.binop_prod_rule.get(prod_head)
		assert emit_inst is not None 

		result_type = self.inst_out_type.get(emit_inst)
		self.temp_var_info(result_type, node)

		if emit_inst == 'axpby_frac':
			""" fraction buffer filled flag """
			self.add_axpby_frac(node.left.name, node.right.name)
			node.axpby_frac_flag = True
		elif emit_inst == 'axpby':
			if not hasattr(node.left, 'axpby_frac_flag'):	
				fill_var_name = self.const_fill_info(1.0)
				self.add_axpby_frac(fill_var_name, node.left.name)

			if not hasattr(node.right, 'axpby_frac_flag'):	
				if node.op == '-':
					fill_scalar = -1.0
				else:
					fill_scalar = 1.0
				fill_var_name = self.const_fill_info(fill_scalar)
				self.add_axpby_frac(fill_var_name, node.right.name)

			self.emit_axpby_buffer(node.name, op_type)
		elif emit_inst == 'scalar_op':
			self.insert_ir_table(inst_type=emit_inst, 
							op_type=op_type,
							result=node.name,
							s0=node.left.name,
							s1=node.right.name,)
		else:
			self.insert_ir_table(inst_type=emit_inst, 
							op_type=op_type,
							result=node.name,
							v0=node.left.name,
							v1=node.right.name,)
	def visit_Assignment(self, node):
		""" using the assigment value instead of temp var as name""" 
		assert isinstance(node.rvalue, c_ast.BinaryOp) or isinstance(node.rvalue, c_ast.ID)
		if isinstance(node.rvalue, c_ast.BinaryOp):
			node.rvalue.name = node.lvalue.name
			for c in node:
				self.visit(c)
			if self.axpby_buffer['frac_num'] > 0:
				self.emit_axpby_buffer(node.lvalue.name)

		if isinstance(node.rvalue, c_ast.ID):
			left_type = lookup_var_type(self.symbol_table, node.lvalue)
			right_type = lookup_var_type(self.symbol_table, node.rvalue)
			assert left_type == 'vectorf' and right_type == 'vectorf', print(node.lvalue, node.rvalue)
			fill_var_name = self.const_fill_info(1.0)
			self.add_axpby_frac(fill_var_name, node.rvalue.name)
			self.emit_axpby_buffer(node.lvalue.name)
	def visit_FuncCall(self,node):
		func_name = node.name.name
		arg_list = node.args.exprs
		# id_list = list(map(lambda x: x.name, arg_list))
		id_list = [x.name if hasattr(x, 'name') else x for x in arg_list]

		if func_name == 'calc_norm_inf':
			assert len(id_list) ==  2
			self.insert_ir_table(inst_type=func_name, 
							result=id_list[1],
							v0=id_list[0])
		if func_name == 'select_max':
			assert len(id_list) ==  3
			self.insert_ir_table(inst_type='scalar_op', 
							op_type=func_name,
							s0=id_list[1],
							s1=id_list[2],
							result=id_list[0])
		if func_name == 'select_min':
			assert len(id_list) ==  3
			self.insert_ir_table(inst_type='scalar_op', 
							op_type=func_name,
							s0=id_list[1],
							s1=id_list[2],
							result=id_list[0])
		if func_name == 'c_sqrt':
			assert len(id_list) == 2
			self.insert_ir_table(inst_type='scalar_op', 
								 op_type=func_name,
								 s0=id_list[1],
								 s1=id_list[1],
								 result=id_list[0])

		if func_name == 'ew_prod':
			assert len(id_list) == 3
			""" lhs to cvb """
			self.insert_ir_table(inst_type='nop') 
			""" ew_prod using omega network """
			self.insert_ir_table(inst_type='axpby', 
								 op_type=func_name,
								 v0=id_list[1],
								 v1=id_list[2],
								 result=id_list[0])
			""" result to rhs """
			self.insert_ir_table(inst_type='nop') 

		if func_name == 'ew_reciprocal':
			assert len(id_list) == 2
			self.insert_ir_table(inst_type='axpby-reci', 
								 op_type=func_name,
								 v0=id_list[1],
								 result=id_list[0])

		if func_name == 'set_scalar_conditional':
			assert len(id_list) == 5
			self.insert_ir_table(inst_type='axpby-setcond', 
								 op_type=func_name,
								 result=id_list[0],
								 v0=id_list[1],
								 v1=id_list[2],
								 s0=id_list[3],
								 s1=id_list[4])

		if func_name =='omega_net': 
			""" v0: config name
				s0: offset    
			"""
			assert len(id_list) == 2
			self.insert_ir_table(inst_type=func_name, 
								 v0=id_list[0],
								 s0=id_list[1])

		if func_name == 'load_cvb' or\
			  func_name == 'cvb_write':
			assert len(id_list) == 3
			""" v0: vector name
				s0: offset
				s1: pack_size 
			"""
			self.insert_ir_table(inst_type=func_name, 
								 v0=id_list[0],
								 s0=id_list[1],
								 s1=id_list[2])

		if func_name == 'gather_scatter':
			assert len(id_list) == 1
			self.insert_ir_table(inst_type=func_name, 
								 v0=id_list[0])

		if func_name == 'dot':
			assert len(id_list) == 3
			self.insert_ir_table(inst_type='nop') 

			self.insert_ir_table(inst_type=func_name, 
								 result=id_list[0],
								 v0=id_list[1],
								 v1=id_list[2],
								 length=1)
			""" cvb accumulate """
			self.insert_ir_table(inst_type='nop') 

			""" result to RF """
			self.insert_ir_table(inst_type='nop') 

	def visit_If(self, node):
		self.visit(node.cond)
		""" -> jump to if_false_start if cond 0 """
		self.insert_ir_table(inst_type='branch', 
						op_type='if_false_start',
						s0=node.cond.name)
		self.visit(node.iftrue)
		""" -> uncondition jump to if_false_end """
		self.insert_ir_table(inst_type='branch', 
						op_type='if_false_end')
		if_false_start= self.current_inst_addr()
		if node.iffalse is not None:
			self.visit(node.iffalse)
		if_false_end = self.current_inst_addr()

		df = self.ir_table
		marked_start = False
		marked_end = False
		for idx in reversed(range(len(df))):
			if df.at[idx, 'inst_type'] == 'branch' and\
				  df.at[idx, 'result'] is None:

				if df.at[idx, 'op_type'] == 'if_false_start': 
					df.at[idx, 'result'] = if_false_start
					marked_start = True

				if df.at[idx, 'op_type'] == 'if_false_end': 
					df.at[idx, 'result'] = if_false_end
					marked_end = True

				if marked_end and marked_start:
					break

		assert marked_end and marked_start,\
			print(marked_start, marked_end)

	def visit_While(self, node):
		""" -> jump to while_end if cond 0 """
		while_start = self.current_inst_addr()
		self.visit(node.cond)
		self.insert_ir_table(inst_type='branch', 
						op_type='while_end',
						s0=node.cond.name)
		self.visit(node.stmt)
		""" -> uncondition jump to while_start """
		self.insert_ir_table(inst_type='branch', 
						op_type='while_start')
		while_end = self.current_inst_addr()

		df = self.ir_table
		marked_start = False
		marked_end = False
		for idx in reversed(range(len(df))):
			if df.at[idx, 'inst_type'] == 'branch' and\
				  df.at[idx, 'result'] is None:
			
				if df.at[idx, 'op_type'] == 'while_start': 
					df.at[idx, 'result'] = while_start 
					marked_start = True

				if df.at[idx, 'op_type'] == 'while_end': 
					df.at[idx, 'result'] = while_end
					marked_end = True

				if marked_end and marked_start:
					break

		assert marked_end and marked_start,\
			print(marked_start, marked_end)

	def visit_Compound(self, node):
		for stmts in node.block_items:
			self.visit(stmts)

	def current_inst_addr(self):
		current_addr = self.ir_table['length'].sum()
		return current_addr

	def init_pass(self):
		for _, row in self.symbol_table.iterrows():
			""" add all vectors to bp graph""" 
			if row['type'] == 'vectorf':
				if row['id'] not in self.bp_graph_nodes:
					self.bp_graph_nodes.append(row['id'])

			if row['type'] == 'float':
				self.reg_onchip.append(row['id'])

			if row['type'] == 'omegaconf':
				self.omegaconf_list.append(row['id'])

	def oneside_pass(self):
		""" Put everything on RHS 
		If no vector op, skip this pass """
		graph_size = len(self.bp_graph_nodes)	
		if graph_size == 0:
			return
		for vec_id in self.bp_graph_nodes:
			self.rhs_hbm.append(vec_id)

	def codegen_pass(self):
		""" Generate register init value """
		for idx, item in enumerate(self.reg_onchip):
			item_value = self.lookup_id_value(item)
			df_insert_row(self.reg_layout, [item, item_value])
			if item_value is not None:
				self.register_file[idx]=item_value
		for _, row in self.ir_table.iterrows():
			if row['inst_type'] == 'dot':
				self.lhs_to_cvb(row['v0'])

				assert row['v1'] in self.rhs_hbm, print(row['v1'])
				src_rhs_addr = self.vec_addr(row['v1'])
				self.add_omega_inst(hbm_addr=src_rhs_addr*self.unified_vec_pack_len,
						 			hbm_pack_size=self.unified_vec_pack_len,
									l2_opcode=1,
									cvb_offset=0)

				""" omega accumulate the ew prod """
				config_offset = self.lookup_omegaconf('vector_sum')
				self.add_omega_inst(hbm_addr=self.cu_arg_info[data_pack_num*config_offset + 0],
									hbm_pack_size=self.cu_arg_info[data_pack_num*config_offset + 2] ,
									l2_opcode=2,
									cvb_offset=0)

				""" write accumulate result to result_addr """
				result_addr = self.lookup_reg_addr(row['result'])
				self.add_rf_write_inst(dst_rf_addr=result_addr,
						   				bank_addr=0,
										loc_addr=self.unified_vec_pack_len)

			if row['inst_type'] == 'axpby':
				assert row['v1'] in self.rhs_hbm or row['v1'] is None

				if row['s0'] is None:
					alpha_addr = self.lookup_reg_addr('const_zero')
				else:
					alpha_addr = self.lookup_reg_addr(row['s0'])

				if row['s1'] is None:
					beta_addr = self.lookup_reg_addr('const_zero')
				else:
					beta_addr = self.lookup_reg_addr(row['s1'])

				src_rhs_addr = self.vec_addr(row['v1'])

				assert row['result'] in self.rhs_hbm 
				dst_addr = self.vec_addr(row['result'])

				assert row['op_type'] in self.axpby_type_dict

				if row['op_type'] == 'ew_prod':
					self.lhs_to_cvb(row['v0'])
					self.add_omega_inst(hbm_addr=src_rhs_addr*self.unified_vec_pack_len,
						 				hbm_pack_size=self.unified_vec_pack_len,
										l2_opcode=1,
										cvb_offset=0)
					self.result_to_rhs(row['result'])
				elif row['op_type'] == 'linear':
					assert row['v0'] in self.rhs_hbm, print(row['v0'])
					self.add_cvb_broadcast_inst(
						src_rf_addr=alpha_addr,
						loc_addr=self.unified_vec_pack_len)
					src_lhs_addr=self.vec_addr(row['v0'])
					self.add_omega_inst(hbm_addr=src_lhs_addr*self.unified_vec_pack_len,
									hbm_pack_size=self.unified_vec_pack_len,
									l2_opcode=3,
									cvb_offset=0)
					if row['v1'] is not None:
						self.add_cvb_broadcast_inst(
							src_rf_addr=beta_addr,
							loc_addr=2*self.unified_vec_pack_len)
						self.add_omega_inst(hbm_addr=src_rhs_addr*self.unified_vec_pack_len,
											hbm_pack_size=self.unified_vec_pack_len,
											l2_opcode=3,
											cvb_offset=self.unified_vec_pack_len)
						""" add 2 vectors on cvb """
						self.add_omega_inst(hbm_addr=0,
											hbm_pack_size=self.unified_vec_pack_len,
											l2_opcode=4,
											input_mode=1,
											operator_mode=1,
											cvb_offset=0)

					self.result_to_rhs(row['result'])

				else:
					self.lhs_to_cvb(row['v0'])
					op_type = self.axpby_type_dict.get(row['op_type'])
					self.add_axpby_inst(alpha_addr=alpha_addr,
										beta_addr=beta_addr,
										src_tetris_addr=0,
										src_vf_addr=src_rhs_addr,
										dst_addr=dst_addr,
										op_type=op_type,
										gamma_addr=0,
										pack_size=self.unified_vec_pack_len)

			if row['inst_type'] == 'axpby-reci':
				assert row['v0'] in self.rhs_hbm 
				assert row['result'] in self.rhs_hbm 

				reci_addr = self.vec_addr(row['v0'])
				dst_addr = self.vec_addr(row['result'])

				assert row['op_type'] in self.axpby_type_dict
				op_type = self.axpby_type_dict.get(row['op_type'])
				self.add_axpby_inst(alpha_addr=0,
									beta_addr=0,
									src_tetris_addr=reci_addr,
									src_vf_addr=reci_addr,
									dst_addr=dst_addr,
									op_type=op_type,
									gamma_addr=0,
									pack_size=self.unified_vec_pack_len)

			if row['inst_type'] == 'axpby-setcond':
				assert row['v0'] in self.rhs_hbm 
				assert row['result'] in self.rhs_hbm 

				cond_addr = self.vec_addr(row['v0'])
				dst_addr = self.vec_addr(row['result'])

				assert row['v1'] in self.reg_onchip
				alpha_addr = self.lookup_reg_addr(row['v1'])
				assert row['s0'] in self.reg_onchip
				beta_addr = self.lookup_reg_addr(row['s0'])
				assert row['s1'] in self.reg_onchip
				gamma_addr = self.lookup_reg_addr(row['s1'])

				assert row['op_type'] in self.axpby_type_dict
				op_type = self.axpby_type_dict.get(row['op_type'])
				self.add_axpby_inst(alpha_addr=alpha_addr,
									beta_addr=beta_addr,
									src_tetris_addr=cond_addr,
									src_vf_addr=cond_addr,
									dst_addr=dst_addr,
									op_type=op_type,
									gamma_addr=gamma_addr,
									pack_size=self.unified_vec_pack_len)

			if row['inst_type'] == 'calc_norm_inf':
				assert row['v0'] in self.rhs_hbm
				src_addr = self.vec_addr(row['v0'])
				assert row['result'] in self.reg_onchip
				dst_addr = self.lookup_reg_addr(row['result'])
				self.add_norm_inf_inst(src_addr, 
											  dst_addr,
											  0,
											  self.unified_vec_pack_len)

			if row['inst_type'] == 'scalar_op':
				assert row['s0'] in self.reg_onchip and\
					row['s1'] in self.reg_onchip and\
					row['result'] in self.reg_onchip
				src_0_reg = self.lookup_reg_addr(row['s0'])
				src_1_reg = self.lookup_reg_addr(row['s1'])
				dst_reg = self.lookup_reg_addr(row['result'])
				imme_flag=0

				assert row['op_type'] in self.scalar_op_type_dict
				op_type = self.scalar_op_type_dict.get(row['op_type'])
				self.add_scalar_op_inst(src_0_reg, 
											   src_1_reg,
											   op_type,
											   dst_reg,
											   imme_flag)

			if row['inst_type'] == 'branch':
				jump_address= row['result']
				if row['s0'] is None: # the case of unconditioned jump
					self.add_branch_inst(src_0_reg=0, 
						 				force_jump=1, 
						 				jump_address=jump_address)
				else:
					assert row['s0'] in self.reg_onchip
					src_0_reg = self.lookup_reg_addr(row['s0'])
					self.add_branch_inst(src_0_reg=src_0_reg, 
										force_jump=0, 
										jump_address=jump_address)

			if row['inst_type'] == 'load_cvb':
				assert row['v0'] in self.rhs_hbm
				src_addr = self.vec_addr(row['v0'])

				cvb_offset = self.offset_translate(row['s0'])
				pack_size = self.offset_translate(row['s1'])

				self.add_load_cvb_inst(src_addr=src_addr,
									src_sel_lhs=0,
									cvb_offset=cvb_offset,
									pack_size=pack_size)

			if row['inst_type'] == 'omega_net':
				config_offset = self.lookup_omegaconf(row['v0'])
				cvb_offset = self.offset_translate(row['s0'])
				self.add_omega_inst(hbm_addr=self.cu_arg_info[data_pack_num*config_offset + 0],
									hbm_pack_size=self.cu_arg_info[data_pack_num*config_offset + 2] ,
									l2_opcode=0,
									cvb_offset=cvb_offset)

			if row['inst_type'] == 'cvb_write':
				assert row['v0'] in self.rhs_hbm
				dst_addr = self.vec_addr(row['v0'])

				cvb_offset = self.offset_translate(row['s0'])
				pack_size = self.offset_translate(row['s1'])
				cut_off_bank = self.cut_off_translate(row['s1'])

				self.add_cvb_write_inst(dst_addr=dst_addr,
										dst_sel_lhs=0,
										cvb_offset=cvb_offset,
										pack_size=pack_size,
										cut_off_bank=cut_off_bank)


	def vec_addr(self, row_item):
		assert row_item in self.rhs_hbm or row_item is None,\
			print(row_item, self.rhs_hbm)
		if row_item is None:
			lr_addr = 0
		elif row_item in self.rhs_hbm:
			lr_addr = self.rhs_hbm.index(row_item)
		return lr_addr 

	def lookup_reg_addr(self, row_item):
		if row_item in self.reg_onchip:
			return self.reg_onchip.index(row_item)
		else:
			return 0

	def lookup_omegaconf(self, row_item):
		assert row_item in self.omegaconf_list, print("omegaconf {} not found!".format(row_item))
		return self.omegaconf_list.index(row_item)

	def lookup_cvb_offset(self, var_name):
		offset_key = var_name + '_cvb_offset'
		if offset_key in self.cu_dict:
			cvb_offset = self.cu_dict[var_name + '_cvb_offset']
		else:
			cvb_offset = 0
		return cvb_offset

	def lookup_id_value(self,  id):
		df = self.symbol_table
		id_record = df.loc[df['id'] == id]
		""" the ID should be unique """
		assert len(id_record) <= 1, print(id)
		if len(id_record) == 1:
			id_value = id_record.iloc[0]['const_value']
		else:
			id_value = 0.0
		return id_value

	def program_gen(self, src_file):
		""" Compile the C Source """
		ast = parse_file(src_file, use_cpp=False)
		main_stmts = ast.ext[0].body.children()
		for _, stmt in enumerate(main_stmts):
			self.visit(stmt[1])

		self.init_pass()
		self.ir_table['addr'] = self.ir_table['length'].cumsum()-1
		self.ir_table.to_csv('./temp/ir_table.csv', index=False)
		self.symbol_table.to_csv('./temp/symbol_table.csv', index=False)

		self.oneside_pass()

		""" Config omega network programs """
		self.nnz_mem_offset = len(self.rhs_hbm)*self.unified_vec_pack_len
		for config_item in self.omegaconf_list[1:]:
			self.dynamic_config(config_item) 

		self.codegen_pass()
		self.reg_layout.to_csv('./temp/reg_layout.csv', index=True)

		""" make sure # of inst % 4 == 0"""
		inst_total_num = self.cu_inst_num
		inst_rom_align_padding = 4 - inst_total_num % 4
		assert len(self.ir_table)==inst_total_num, print('inst num', inst_total_num, len(self.ir_table))
		for _ in range(inst_rom_align_padding-1):
			self.add_inst([self.inst_halt, 0, 0, 0])
		self.add_inst([self.inst_halt, 0, 0, 0], program_end=True)
		assert self.cu_inst_num < 2048, print("Exceed Inst ROM")

		""" matrix info pack 0 is reserved """
		self.cu_arg_info[data_pack_num*0 + 0] = self.unified_vec_pack_len
		self.cu_arg_info[data_pack_num*0 + 1] = self.sol_vec_pack_len
		self.cu_arg_info[data_pack_num*0 + 2] = self.register_file_offset

		""" Add omega instruction info into indice HBM """
		self.add_vector_uint32(self.cu_arg_info, 
						 record_stage_size=True)

	def init_values(self):
		""" Load the register file"""
		rf_pack_size = len(self.register_file)//data_pack_num
		vector_rf = np.zeros(self.isca_c*rf_pack_size)
		for i in range(rf_pack_size):
			vector_rf[i*self.isca_c:i*self.isca_c + data_pack_num]=self.register_file[i*data_pack_num:(i+1)*data_pack_num]
		self.cu_dict['register_file'] = vector_rf

		vector_zero_padded = np.zeros(self.isca_c)

		for vec_item in self.rhs_hbm:
			init_item = self.cu_dict.get(vec_item, 
									vector_zero_padded)
			self.add_vector_rhs_mem(init_item)

		""" Check reg and vec addr space overflow """
		assert len(self.reg_onchip) < self.RegFileSize
		assert len(self.rhs_hbm) < 64
