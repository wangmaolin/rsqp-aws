import scipy
import numpy as np
import logging

import scipy.sparse
from mib_sched import DragonC10

def check_empty_rows(mat):
	nnz_along_row = mat.getnnz(axis=1)
	empty_rows = np.sum(nnz_along_row == 0)
	assert empty_rows == 0, logging.debug("Empty rows %d", empty_rows)

def save_to_debug(vec, vec_name):
	vec.astype(np.float32).tofile('./temp/'+vec_name)

def Setup_SpMV(cu_dict, qp_problem, scalars):
	""" Deal with empty rows in matrices """
	check_empty_rows(qp_problem['A'])
	check_empty_rows(qp_problem['A'].T)

	compileP = DragonC10(qp_problem['P'].tocsr(), 
					 iscaC=scalars['isca_c'], 
					 readOffset=scalars['pdim_max'], 
					 pdimMax=scalars['pdim_max'],
					 plotName='P')
	cu_dict['P_multiply'] = compileP.top_pass()

	compileA = DragonC10(qp_problem['A'].tocsr(), 
					 iscaC=scalars['isca_c'], 
					 readOffset=scalars['pdim_max'], 
					 pdimMax=scalars['pdim_max'],
					 plotName='A')
	cu_dict['A_multiply'] = compileA.top_pass()

	compileAtrans = DragonC10(qp_problem['A'].T.tocsr(), 
					 iscaC=scalars['isca_c'], 
					 readOffset=scalars['pdim_max'], 
					 pdimMax=scalars['pdim_max'],
					 plotName='Atrans')
	cu_dict['At_multiply'] = compileAtrans.top_pass()

def Setup_Constr(cu_dict, qp_problem):
	OSQP_RHO_TOL = 1e-4
	OSQP_INFVAL_SCALED = 1e26
	loose_bounds = np.logical_and(qp_problem['l']<-OSQP_INFVAL_SCALED,
							   qp_problem['u']>OSQP_INFVAL_SCALED)
	loose_bounds = loose_bounds.astype(np.float32) * -1
	eq_constr = (qp_problem['u'] - qp_problem['l']) < OSQP_RHO_TOL
	eq_constr = eq_constr.astype(np.float32) * 1
	combined_constr = loose_bounds + eq_constr
	cu_dict['constr_type'] = np.where(combined_constr == 0.0, 2.0, combined_constr)

	cu_dict['data_l'] = qp_problem['l']
	cu_dict['data_u'] = qp_problem['u']
	cu_dict['data_q'] = qp_problem['q']

def osqp_indirect(cu_dict, qp_problem, scalars):
	""" Compute preconditioner for PCG"""
	PsI = qp_problem['P'] +\
		scalars['sigma'] * scipy.sparse.identity(qp_problem['P'].shape[0])
	diag_PsI = PsI.diagonal()
	AtA = qp_problem['A'].transpose() * qp_problem['A']
	diag_AtA= AtA.diagonal()
	cu_dict['diag_PsI'] = diag_PsI
	cu_dict['diag_AtA'] = diag_AtA

	Setup_SpMV(cu_dict, qp_problem, scalars)
	Setup_Constr(cu_dict, qp_problem)

	""" Debug PCG """
	# Check_PCG(cu_dict, qp_problem, scalars, diag_PsI, diag_AtA)
	# zero_vec = np.zeros(scalars['pdim_n']).astype(np.float32)
	# save_to_debug(zero_vec, 'test_out')

def ut_spmv(cu_dict, qp_problem, scalars):
	""" Int Matrix 
	debug_nz_cnt=17
	diagElements=np.arange(1, scalars['pdim_n']+1)
	SpMat = scipy.sparse.diags(diagElements).tocsr()
	for rowItem in range(2):
		SpMat[rowItem, :debug_nz_cnt]=np.ones(debug_nz_cnt)
	spmv_in = np.ones(scalars['pdim_n']).astype(np.float32)
	"""

	""" Lower Triangular Matrix 
	SpMat = scipy.sparse.csr_matrix((scalars['pdim_n'], scalars['pdim_n']))
	for rowItem in range(scalars['pdim_n']):
		SpMat[rowItem, :rowItem]=np.ones(rowItem)
	spmv_in = np.ones(scalars['pdim_n']).astype(np.float32)
	"""

	""" All dense block
	SpMat = scipy.sparse.csr_matrix((scalars['pdim_n'], scalars['pdim_n']))
	blockSize=242 # error 14
	for rowItem in range(1):
		SpMat[rowItem, :blockSize]=np.ones(blockSize)*0.1
	spmv_in = np.ones(scalars['pdim_n']).astype(np.float32)
	"""

	""" Random matrix
	SpMat = scipy.sparse.random(scalars['pdim_n'], 
					  scalars['pdim_m'], 
					  density=0.1, 
					  format='csr')
	spmv_in = np.random.rand(scalars['pdim_m']).astype(np.float32)
	"""

	# """ SVM At mul wrong 
	SpMat = qp_problem['A'].T
	spmv_in = np.random.rand(scalars['ori_dim_m']).astype(np.float32)
	# """

	# SpMat = qp_problem['A']
	# spmv_in = np.random.rand(scalars['ori_dim_n']).astype(np.float32)

	test_out = SpMat @ spmv_in
	save_to_debug(test_out, 'test_out')
	cu_dict['spmv_in'] = spmv_in
	cu_dict['zero_vec'] = np.zeros(scalars['pdim_max']).astype(np.float32)

	""" Functional style scheduling """
	o3issue = DragonC10(SpMat.tocsr(), 
					 iscaC=scalars['isca_c'], 
					 readOffset=scalars['pdim_max'], 
					 pdimMax=scalars['pdim_max'],
					 plotName='ut')
	cu_dict['SpMat'] = o3issue.top_pass()

def ut_vecop(cu_dict, qp_problem, scalars):
	# vec_a = np.arange(scalars['pdim_n']) + 1
	# vec_b = np.arange(scalars['pdim_n']) + 2
	# vec_c = np.arange(scalars['pdim_n']) + 3

	vec_a = np.random.rand(scalars['pdim_n']).astype(np.float32)
	vec_b = np.random.rand(scalars['pdim_n']).astype(np.float32)
	vec_c = np.random.rand(scalars['pdim_n']).astype(np.float32)

	# vec_a = np.ones(scalars['pdim_n'])*3
	# vec_b = np.ones(scalars['pdim_n'])*2
	# vec_c = np.ones(scalars['pdim_n'])

	test_out = vec_a - 2*vec_b + 2*vec_c
	print('dot result: ', np.dot(vec_a, vec_b))

	cu_dict['zero_vec'] = np.zeros(scalars['pdim_max']).astype(np.float32)
	cu_dict['vec_a'] = vec_a
	cu_dict['vec_b'] = vec_b
	cu_dict['vec_c'] = vec_c

	save_to_debug(test_out, 'test_out')

def Check_PCG(cu_dict, qp_problem, scalars, diag_PsI, diag_AtA):
	""" Compute preconditioner for PCG"""
	precond = np.reciprocal(diag_PsI + scalars['rho']*diag_AtA)
	pcg_computed_rhs = np.random.rand(scalars['ori_dim_n']).astype(np.float32)
	cu_dict['pcg_computed_rhs'] = pcg_computed_rhs

	pcg_sol = np.random.rand(scalars['ori_dim_n']).astype(np.float32)
	""" Init PCG sol"""
	cu_dict['pcg_sol'] = pcg_sol

	KKT_mat = qp_problem['P'] + scalars['sigma'] * scipy.sparse.identity(scalars['ori_dim_n']) +\
			scalars['rho']*qp_problem['A'].T*qp_problem['A']

	pcg_eps=1e-4

	np_sol = scipy.sparse.linalg.spsolve(KKT_mat, pcg_computed_rhs)
	pcg_Kx = KKT_mat @ pcg_sol
	pcg_res = pcg_Kx - pcg_computed_rhs
	pcg_d = np.multiply(precond, pcg_res)
	save_to_debug(pcg_d, 'test_out')
	norm_rhs=np.linalg.norm(pcg_computed_rhs, ord=np.inf)
	pcg_converge_eps = pcg_eps * norm_rhs
	r_dot_y = np.dot(pcg_res, pcg_d)
	logging.debug('r_dot_y %f', r_dot_y)
	norm_pcg_res =np.linalg.norm(pcg_res, ord=np.inf)
	pcg_mu = 0.0
	pcg_max_iter = 1
	pcg_iter = 0
	pcg_p = 0 * pcg_d
	while pcg_iter < pcg_max_iter:
		pcg_p = pcg_mu * pcg_p - pcg_d
		pcg_Kx = KKT_mat @ pcg_p
		p_dot_Kp = np.dot(pcg_p, pcg_Kx)
		# save_to_debug(pcg_Kx, 'pcg_Kx')

		pcg_iter += 1

	# pcg_sol = np.zeros(P.shape[0])
	# pcg_sol = pcg_computed_rhs
	# pcg_Kx = KKT_mat @ pcg_computed_rhs
	# + sigma * pcg_computed_rhs+rho*A.transpose()*A*pcg_computed_rhs
