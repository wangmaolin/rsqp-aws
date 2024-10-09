import torch
import numpy as np
import sys
sys.path.append('../aws')

from benchmark_gen import problem_instance_gen
import warnings
warnings.filterwarnings("ignore")

import osqp
import matplotlib.pyplot as plt

def calc_norm_inf(vec):
    return torch.norm(vec, p=float('inf'))

def ew_reciprocal(vec):
    return 1.0 / vec

def dot(vec1, vec2):
    return torch.dot(vec1, vec2)

def select_max(vec1, vec2):
    return torch.max(vec1, vec2)

def c_sqrt(a):
    return torch.sqrt(a)

def mp_spmv(mat, vec):
    """ Mixed precision SpMV """
    """ torch.float16 doesn't support sparse matrix input yet """
    # return torch.matmul(mat, vec)
    return torch.matmul(mat.to_dense(), vec)

class osqpTorch:
    """ Research of using mixed precision in the OSQP algorithm,
     Direct translation from a simplified version of the algorithm in ../aws/osqp_indirect.c """
    def __init__(self, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype

    def setup(self, P, q, A, l, u):
        """ Dimension of the solution vector"""
        self.dimN = P.shape[0]
        self.dimM = A.shape[0]

        self.admm_iters = 0
        self.max_iter = 4000
        self.rho = 0.1
        self.sigma = 1e-6
        self.alpha = 1.6
        self.one_minus_alpha = 1 - self.alpha
        self.settings_eps_abs = 1e-3
        self.settings_eps_rel = 1e-3
        self.pcg_max_iter = 20
        self.pcg_eps = 1e-4
        self.check_termination = 10
        """ Matrix """ 
        self.sparse_P = self.create_matrix(P)
        self.sparse_A = self.create_matrix(A)
        self.diag_PsI = self.create_vector(
            np_init = np.array(P.diagonal() + self.sigma))
        self.diag_AtA = self.create_vector(
            np_init = np.array((A.T@A).diagonal()))

        """ The optimal solution """
        self.work_x = self.create_vector(self.dimN)
        self.data_l = self.create_vector(np_init=l)
        self.data_u = self.create_vector(np_init=u)
        self.data_q = self.create_vector(np_init=q)

    def create_vector(self, size=0, np_init=None):
        if np_init is None:
            return torch.zeros(size, device=self.device, dtype=self.dtype)
        else:
            return torch.from_numpy(np_init).to(self.dtype).to(self.device)

    def create_matrix(self, csc_matrix):
        coo_matrix = csc_matrix.tocoo()
        values = torch.tensor(coo_matrix.data, 
                              dtype=self.dtype)
        indices = torch.tensor(
            [coo_matrix.row, coo_matrix.col], 
            dtype=torch.int64)

        sparse_tensor = torch.sparse_coo_tensor(indices, values, coo_matrix.shape).to(self.device)
        return sparse_tensor


    def KKT_mul(self, vec):
        """ Multiply the KKT matrix P+ sigma*I + rho*AtA"""
        """ first Px0 """
        pcg_P_mul_x = mp_spmv(self.sparse_P, vec)
        """ AtAx0 """
        A_mul_x = mp_spmv(self.sparse_A, vec)
        pcg_At_mul_y = mp_spmv(self.sparse_A.T,
                                            A_mul_x)
        pcg_Kx = pcg_P_mul_x +\
            self.rho*pcg_At_mul_y +\
                self.sigma*vec
        return pcg_Kx

    def solve(self):
        """ init vectors """
        work_z = self.create_vector(self.dimM)
        work_y = self.create_vector(self.dimM)
        pcg_p = self.create_vector(self.dimN)
        pcg_sol = self.create_vector(self.dimN)

        """ osqp_update_rho() in osqp_api.c """
        norm_q = calc_norm_inf(self.data_q)
        rho_recipical = 1.0 / self.rho
        self.total_pcg_iters = 0

        """ update preconditioner """
        precond = self.diag_PsI + self.rho * self.diag_AtA
        precond = ew_reciprocal(precond)

        """ ----- osqp_solve() in osqp_api.c----- """
        termination = False
        while self.admm_iters < self.max_iter and not termination:
            prev_x = self.work_x.clone()
            prev_z = work_z.clone()

            """ ----- update_xz_tilde() in osqp_api.c ------ """
            """ compute_rhs() in auxil.c """
            xtilde_view = self.sigma * prev_x - self.data_q
            ztilde_view = rho_recipical * work_y
            ztilde_view = prev_z - ztilde_view

            """ PCG preparation """
            """ 0-4 A^T r2 """
            pcg_At_mul_y = mp_spmv(self.sparse_A.T, 
                                        ztilde_view)
            """ 0-5 rhs = r1 + rho*At r2 """
            pcg_computed_rhs = xtilde_view +\
                  self.rho*pcg_At_mul_y 
            """ KKT multiply x0 """
            pcg_Kx = self.KKT_mul(pcg_sol)
            """ 13 r0 = Kx0 - rhs """
            pcg_res = pcg_Kx - pcg_computed_rhs
            """ 14. compute d0 = M^-1*r0 """
            pcg_d = precond * pcg_res
            """ 15. inf norm of rhs """
            norm_rhs = calc_norm_inf(pcg_computed_rhs)
            """ 16. epsilon times |rhs| """
            pcg_converge_eps = self.pcg_eps * norm_rhs
            """ 17. r0 dot d0 """
            r_dot_y = dot(pcg_res, pcg_d)
            """ if pcg_res is 0, skip pcg loop """
            norm_pcg_res = calc_norm_inf(pcg_res)
            """ reset pcg states """
            pcg_mu = 0
            pcg_iters = 0
            pcg_termination = False
            if norm_pcg_res<pcg_converge_eps or\
                  norm_rhs < self.pcg_eps:
                pcg_termination = True
            """ PCG loop """
            while pcg_iters < self.pcg_max_iter and\
                  not pcg_termination:
                """ 18. p(k+1) = -d(k+1) + mu p(k) """
                pcg_p = pcg_mu*pcg_p - pcg_d
                """ 19. Kp0 """
                pcg_Kx = self.KKT_mul(pcg_p)
                """ 30. p dot Kp0 """
                p_dot_Kp = dot(pcg_p, pcg_Kx)
                """ 31. lambda = r_dot_y/p_dot_Kp """
                pcg_lambda = r_dot_y/p_dot_Kp
                """ 32 x(k+1)=x(k)+lambda p(k) """
                pcg_sol = pcg_sol + pcg_lambda * pcg_p
                """ 33 r(k+1)=x(k)+lambda Kp """
                pcg_res = pcg_res + pcg_lambda*pcg_Kx
                """ 34 |res| """
                norm_pcg_res = calc_norm_inf(pcg_res)
                """ 35 d = M^-1 r """
                pcg_d = precond * pcg_res
                """ 36 save r_dot_y """
                prev_r_dot_y = r_dot_y.clone()
                """ 37 dot r(k+1) d(k+1) """
                r_dot_y = dot(pcg_res, pcg_d)
                """ 38 compute mu """
                pcg_mu = r_dot_y/prev_r_dot_y
                """ exit if PCG converged """
                if norm_pcg_res<pcg_converge_eps:
                    pcg_termination = True

                pcg_iters += 1
            xtilde_view = pcg_sol

            self.total_pcg_iters = self.total_pcg_iters + pcg_iters
            """ ztilde_view = mat_A * xtilde_view """
            ztilde_view = mp_spmv(self.sparse_A, xtilde_view)
            """ ----- update_x() in osqp_api.c ------ """
            self.work_x = self.alpha * xtilde_view +\
                  self.one_minus_alpha * prev_x
            delta_x = self.work_x - prev_x
            """ ----- update_z() in osqp_api.c ------ """
            alpha_ztilde = self.alpha * ztilde_view + self.one_minus_alpha * prev_z
            work_z = alpha_ztilde + rho_recipical * work_y
            work_z = torch.clamp(work_z, self.data_l, self.data_u)
            """ ----- update_y() in osqp_api.c ------ """
            delta_y = alpha_ztilde - work_z
            work_y = work_y + self.rho * delta_y

            """ ------ update_info() in auxil.c ----- """
            if self.admm_iters % self.check_termination == 0:
                """ compute prime residual """
                A_mul_x = mp_spmv(self.sparse_A, self.work_x)
                prev_z = A_mul_x - work_z
                self.prim_res = calc_norm_inf(prev_z)
                """ compute dual res """
                P_mul_x = mp_spmv(self.sparse_P, self.work_x)
                prev_x = self.data_q + P_mul_x
                At_mul_y = mp_spmv(self.sparse_A.T, work_y)
                prev_x = At_mul_y + prev_x
                self.dual_res = calc_norm_inf(prev_x)
                """ compute_prim_tol """
                norm_z = calc_norm_inf(work_z)
                norm_Ax = calc_norm_inf(A_mul_x)
                prim_norm = select_max(norm_z, norm_Ax)
                eps_prim = self.settings_eps_rel * prim_norm
                eps_prim = self.settings_eps_abs + eps_prim
                """ compute_dual_tol """
                norm_Aty = calc_norm_inf(At_mul_y)
                temp_rel_eps = select_max(norm_q, norm_Aty)
                norm_Px= calc_norm_inf(P_mul_x)
                dual_norm = select_max(temp_rel_eps, norm_Px)
                eps_dual = self.settings_eps_rel * dual_norm
                eps_dual = self.settings_eps_abs + eps_dual
                """ termination check """
                if self.prim_res < eps_prim and self.dual_res < eps_dual:
                    termination = True
                """ ----- compute_rho_estimate() in auxil.c ----- """
                prim_div = self.prim_res/prim_norm
                dual_div = self.dual_res/dual_norm
                rho_estimate = prim_div/dual_div
                rho_estimate = c_sqrt(rho_estimate)
                """ ----- adapt_rho() in osqp_api.c ----- """
                if rho_estimate > 5.0 or rho_estimate < 0.2:
                    self.rho = rho_estimate 
                    rho_recipical = 1.0/self.rho
                    """ update preconditioner """
                    precond = self.diag_PsI + self.rho * self.diag_AtA
                    precond = ew_reciprocal(precond)

            self.admm_iters += 1

def main():
    """ 100 benchmark problems
        - test_problem_name: Portfolio, Control, SVM, LASSO, Huber
        - dim_idx: 0 to 19
    """
    qp_problem = problem_instance_gen(
        test_problem_name = 'Control', 
        dim_idx = 0)
    """ Solver using mixed precision"""
    prob_mp = osqpTorch(
        # dtype=torch.float32,
        dtype=torch.bfloat16, # works
        # dtype=torch.float16, not work
        # dtype=torch.float8_e5m2fnuz, not work
        # dtype=torch.float8_e4m3fn, not work
        device='cuda')
        # device='cpu')
    prob_mp.setup(
        P=qp_problem['P'], 
        q=qp_problem['q'],
        A=qp_problem['A'],
        l=qp_problem['l'],
        u=qp_problem['u'])
    prob_mp.solve()

    print("prim_res:\t{:.2e}".format(prob_mp.prim_res))
    print("dual_res:\t{:.2e}".format(prob_mp.dual_res))
    print("admm_iters:\t{}".format(prob_mp.admm_iters))
    print("pcg_iters:\t{}".format(prob_mp.total_pcg_iters))
    print("rho:\t{}".format(prob_mp.rho))
    print(len(prob_mp.work_x))

    """ Solve using standard precision"""
    prob=osqp.OSQP()
    prob.setup(
        scaling=0,
        P=qp_problem['P'], 
        q=qp_problem['q'],
        A=qp_problem['A'],
        l=qp_problem['l'],
        u=qp_problem['u'])

    prob_fp32 = prob.solve()
    # print(dir(prob_fp32.info))
    # print(prob_fp32.info.pri_res)
    # print(prob_fp32.info.dua_res)

    plt.plot(prob_mp.work_x.to(torch.float32).cpu(), label='mixed precision')
    plt.plot(prob_fp32.x, label='fp32')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
