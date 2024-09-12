import scipy
import numpy as np
import logging

import scipy.sparse
from mib_sched import MatMulSched
from mib_sched import ShuffleMulSched
from mib_sched import LsolveSched
from mib_sched import UsolveSched
from mib_sched import UpFactorSched
import scipy.sparse as spa
import numpy as np
import qdldl

from qdldl_c_trans import symbolic_factor
from qdldl_c_trans import qdldl_verify

def csr_for_sum(num_rows, num_cols, n):
    """ CSR matrix with one row all 1 for vector sum operation """
    if n < 0 or n >= num_rows:
        raise ValueError("The row index n must be within the range of the matrix dimensions.")

    # Indices of the ones in the nth row
    row_indices = [n] * num_cols
    col_indices = list(range(num_cols))
    data = [1] * num_cols

    # Create the CSR matrix
    csr = spa.csr_matrix((data, (row_indices, col_indices)), 
                         shape=(num_rows, num_cols), 
                         dtype=np.float32)
    return csr

def check_empty_rows(mat):
    nnz_along_row = mat.getnnz(axis=1)
    empty_rows = np.sum(nnz_along_row == 0)
    assert empty_rows == 0, logging.debug("Empty rows %d", empty_rows)

def save_to_debug(vec, vec_name):
    vec.astype(np.float32).tofile('./temp/'+vec_name)

def Setup_LDL_Solve(cu_dict, scalars, csc_L, diag_D, LDU_flag='LDU'):
    """	Forward substitude """
    if 'L' in LDU_flag:
        CompilerLsolve = LsolveSched(
            iscaC=scalars['isca_c'],
            pdimMax=2*scalars['pdim_max'],
            SpMatrix=csc_L.tocsr(),
            plotName='L-solve',
            SkipO3=False
            # SkipO3=True
        )
        cu_dict['LowerSolve'] = CompilerLsolve.top_pass()
        """ O3 Schedule Visualization """
        # CompilerLsolve.plot_dependency()
        # CompilerLsolve.plot_pattern()
        # CompilerLsolve.plot_o3sched()

    if 'D' in LDU_flag:
        natural_order = np.arange(scalars['ori_dim_n']+scalars['ori_dim_m'])
        CompilerDiag = ShuffleMulSched(
            iscaC=scalars['isca_c'],
            pdimMax=2*scalars['pdim_max'],
            srcOrder=natural_order,
            dstOrder=natural_order,
            muls=1.0/diag_D,
            plotName='D-solve',
            SkipO3=False
            # SkipO3=True
        )
        cu_dict['DiagSolve'] = CompilerDiag.top_pass()

    if 'U' in LDU_flag:
        """ the csr form of L.T is csc of L, 
            backward substitude """
        CompilerUsolve = UsolveSched(
            iscaC=scalars['isca_c'],
            pdimMax=2*scalars['pdim_max'],
            SpMatrix=csc_L,
            plotName='U-solve',
            SkipO3=False
            # SkipO3=True
        )
        cu_dict['UpperSolve'] = CompilerUsolve.top_pass()

def Setup_LDL_Perm(cu_dict, scalars, amd_order):
    natural_order = np.arange(scalars['ori_dim_n']+scalars['ori_dim_m'])
    muls=np.ones(scalars['ori_dim_n']+scalars['ori_dim_m']).astype(np.float32)
    CompilerPerm = ShuffleMulSched(
        iscaC=scalars['isca_c'],
        pdimMax=2*scalars['pdim_max'],
        srcOrder=amd_order,
        dstOrder=natural_order,
        muls=muls,
        plotName='Perm')
    cu_dict['Perm'] = CompilerPerm.top_pass()

    CompilerInvPerm = ShuffleMulSched(
        iscaC=scalars['isca_c'],
        pdimMax=2*scalars['pdim_max'],
        srcOrder=natural_order,
        dstOrder=amd_order,
        muls=muls,
        plotName='Perm-inv')
    cu_dict['Perm_inv'] = CompilerInvPerm.top_pass()

def Setup_SpMV(cu_dict, qp_problem, scalars):
    """ Deal with empty rows in matrices """
    check_empty_rows(qp_problem['A'])
    check_empty_rows(qp_problem['A'].T)

    compileP = MatMulSched(qp_problem['P'].tocsr(), 
                     iscaC=scalars['isca_c'], 
                     readOffset=scalars['pdim_max'], 
                     pdimMax=scalars['pdim_max'],
                     plotName='P-mul')
    cu_dict['P_multiply'] = compileP.top_pass()

    # logging.debug('----- Matrix A Start ----')
    compileA = MatMulSched(qp_problem['A'].tocsr(), 
                     iscaC=scalars['isca_c'], 
                     readOffset=scalars['pdim_max'], 
                     pdimMax=scalars['pdim_max'],
                     plotName='A-mul')
    cu_dict['A_multiply'] = compileA.top_pass()
    # logging.debug('----- Matrix A End ----')

    compileAtrans = MatMulSched(qp_problem['A'].T.tocsr(), 
                     iscaC=scalars['isca_c'], 
                     readOffset=scalars['pdim_max'], 
                     pdimMax=scalars['pdim_max'],
                     plotName='At-mul')
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

    dot_len=scalars['pdim_max']
    compileVsum = MatMulSched(csr_for_sum(dot_len+1, dot_len, dot_len), 
                     iscaC=scalars['isca_c'], 
                     readOffset=0, 
                     pdimMax=scalars['pdim_max'],
                     CtrlOnly=True)
    cu_dict['vector_sum'] = compileVsum.top_pass()

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
    # SpMat = qp_problem['A'].T
    # spmv_in = np.random.rand(scalars['ori_dim_m']).astype(np.float32)
    SpMat = qp_problem['P']
    spmv_in = np.random.rand(scalars['ori_dim_n']).astype(np.float32)
    # """

    # SpMat = qp_problem['A']
    # spmv_in = np.random.rand(scalars['ori_dim_n']).astype(np.float32)

    test_out = SpMat @ spmv_in
    save_to_debug(test_out, 'test_out')
    cu_dict['spmv_in'] = spmv_in
    cu_dict['zero_vec'] = np.zeros(scalars['pdim_max']).astype(np.float32)

    """ Functional style scheduling """
    o3issue = MatMulSched(SpMat.tocsr(), 
                     iscaC=scalars['isca_c'], 
                     readOffset=scalars['pdim_max'], 
                     pdimMax=scalars['pdim_max'],
                     plotName='ut')
    cu_dict['SpMat'] = o3issue.top_pass()

def ut_vecop(cu_dict, qp_problem, scalars):
    # vec_a = np.arange(scalars['pdim_n']) + 1
    # vec_b = np.arange(scalars['pdim_n']) + 2
    # vec_c = np.arange(scalars['pdim_n']) + 3
    mul_a = 2.0
    mul_b = -1.0
    mul_c = 1.5
    vec_a = np.random.rand(scalars['pdim_n']).astype(np.float32)
    vec_b = np.random.rand(scalars['pdim_n']).astype(np.float32)
    vec_c = np.random.rand(scalars['pdim_n']).astype(np.float32)

    # vec_a = np.ones(scalars['pdim_n'])*3
    # vec_b = np.ones(scalars['pdim_n'])*2
    # vec_c = np.ones(scalars['pdim_n'])

    # test_out = vec_a * vec_b
    test_out = mul_a*vec_a + mul_b*vec_b + mul_c*vec_c
    print('dot result: ', np.dot(vec_a, vec_b))

    dot_len=scalars['pdim_max']
    compileVsum = MatMulSched(csr_for_sum(dot_len+1, dot_len, dot_len), 
                     iscaC=scalars['isca_c'], 
                     readOffset=0, 
                     pdimMax=scalars['pdim_max'],
                     CtrlOnly=True)
    cu_dict['vector_sum'] = compileVsum.top_pass()

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

def Shift_AMD(amd_order, scalars):
    """ Pad the order """
    inc_mask = (amd_order >= scalars['ori_dim_n'])
    amd_order[inc_mask] += scalars['n_padding']

    permutate_offset = scalars['pdim_m'] + scalars['pdim_n']
    amd_order += permutate_offset
    return amd_order

def ut_perm(cu_dict, qp_problem, scalars):
    natural_order = np.arange(scalars['ori_dim_n']+scalars['ori_dim_m'])
    amd_order=np.random.permutation(natural_order)
    amd_order = Shift_AMD(amd_order, scalars)

    Setup_LDL_Perm(cu_dict, scalars, amd_order)

    in_x = np.arange(0, scalars['ori_dim_n']).astype(np.float32)
    in_z = np.arange(scalars['ori_dim_n'], scalars['ori_dim_m']+scalars['ori_dim_n']).astype(np.float32)

    cu_dict['in_x'] = in_x
    cu_dict['in_z'] = in_z
    save_to_debug(in_x, 'out_x')
    save_to_debug(in_z, 'out_z')

def ut_ldu(cu_dict, qp_problem, scalars):
    factor_dict = symbolic_factor(
        qp_problem['P'], 
        qp_problem['A'], 
        scalars['sigma'], 
        scalars['rho'])

    csc_L = factor_dict['csc_L'] 
    diag_D = factor_dict['diag_D'] 

    """ Unset Random Seed """
    np.random.seed(None)

    concat_in = np.random.rand(csc_L.shape[0]).astype(np.float32)
    cu_dict['in_x']=concat_in[:scalars['pdim_n']]
    cu_dict['in_z']=concat_in[scalars['pdim_n']:]

    test_system = 'L'
    # test_system = 'D'
    # test_system = 'U'

    Setup_LDL_Solve(cu_dict, scalars, csc_L, diag_D, LDU_flag=test_system)

    if test_system == 'L':
        Lsolve = csc_L + scipy.sparse.identity(csc_L.shape[0])
        concat_out = scipy.sparse.linalg.spsolve(Lsolve, concat_in)

    if test_system == 'D':
        Dsolve = scipy.sparse.diags(diag_D) 
        concat_out = scipy.sparse.linalg.spsolve(Dsolve, concat_in)
    
    if test_system == 'U':
        Usolve = csc_L.T + scipy.sparse.identity(csc_L.shape[0])
        concat_out = scipy.sparse.linalg.spsolve(Usolve, concat_in)
    """ cvb_write has cut_off_bank settings, 
        the last few results in out_x are cut off to be 0 """
    save_to_debug(concat_out[:scalars['pdim_n']], 'out_x')
    save_to_debug(concat_out[scalars['pdim_n']:], 'out_z')

def ut_ldl(cu_dict, qp_problem, scalars):
    factor_dict = symbolic_factor(
        qp_problem['P'], 
        qp_problem['A'], 
        scalars['sigma'], 
        scalars['rho'])

    csc_L = factor_dict['csc_L'] 
    diag_D = factor_dict['diag_D'] 
    amd_order = factor_dict['amd_order'] 
    kkt_mat = factor_dict['kkt_mat'] 

    np.random.seed(None)

    amd_order = Shift_AMD(amd_order, scalars)
    Setup_LDL_Perm(cu_dict, scalars, amd_order)

    concat_in = np.random.rand(csc_L.shape[0]).astype(np.float32)
    concat_in[:scalars['ori_dim_n']]=np.fromfile('./temp/in_x',dtype=np.float32)
    concat_in[scalars['ori_dim_n']:]=np.fromfile('./temp/in_z',dtype=np.float32)
    cu_dict['in_x']=concat_in[:scalars['ori_dim_n']]
    cu_dict['in_z']=concat_in[scalars['ori_dim_n']:]

    concat_out=scipy.sparse.linalg.spsolve(kkt_mat, concat_in)

    save_to_debug(concat_out[:scalars['ori_dim_n']], 'out_x')
    save_to_debug(concat_out[scalars['ori_dim_n']:], 'out_z')

    Setup_LDL_Solve(cu_dict, scalars, csc_L, diag_D)

def osqp_direct(cu_dict, qp_problem, scalars):
    factor_dict = symbolic_factor(
        qp_problem['P'], 
        qp_problem['A'], 
        scalars['sigma'], 
        scalars['rho'])

    csc_L = factor_dict['csc_L'] 
    diag_D = factor_dict['diag_D'] 
    amd_order = factor_dict['amd_order'] 

    # check_empty_rows(csc_L.T.tocsr())
    # check_empty_rows(csc_L.tocsr())
    amd_order = Shift_AMD(amd_order, scalars)

    Setup_LDL_Perm(cu_dict, scalars, amd_order)
    Setup_LDL_Solve(cu_dict, scalars, csc_L, diag_D)
    Setup_SpMV(cu_dict, qp_problem, scalars)
    Setup_Constr(cu_dict, qp_problem)

def ut_factor(cu_dict, qp_problem, scalars):
    """
    qdldl_verify( qp_problem['P'], 
        qp_problem['A'], 
        scalars['sigma'], 
        scalars['rho'])
    """

    factor_dict = symbolic_factor(
                qp_problem['P'],
                  qp_problem['A'],
                scalars['sigma'], 
                scalars['rho'])

    CompilerFactor = UpFactorSched(
        iscaC=scalars['isca_c'],
        pdimMax=2*scalars['pdim_max'],
        SpMatrix=factor_dict['perm_KKT'],
        etree=factor_dict['etree'],
        Lnz=factor_dict['Lnz'])

    cu_dict['UpFactor'] = CompilerFactor.top_pass()
