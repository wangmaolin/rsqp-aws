import torch

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

class osqpTorch:
    """ Research of using mixed precision in the OSQP algorithm,
     Direct translation from a simplified version of the algorithm in ../aws/osqp_indirect.c """
    def __init__(self, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype

    def setup(self, P, q, A, l, u):
        """ Dimension of the solution vector"""
        self.dimN = P.shape[0]
        self.dimM = A.shape[1]

        self.admm_iters = 0.0
        self.max_iter = 101.0
        self.rho = 0.1
        self.sigma = 1e-6
        self.alpha = 1.6
        self.one_minus_alpha = 1 - self.alpha
        self.settings_eps_abs = 1e-3
        self.settings_eps_rel = 1e-3
        self.pcg_max_iter = 20.0
        self.pcg_eps = 1e-4
        self.check_termination = 25.0
        """ Matrix """ 
        self.sparse_P = self.create_matrix(P)
        self.sparse_A = self.create_matrix(A)
        self.diag_PsI = self.create_vector(P.diagonal() + self.sigma)
        self.diag_AtA = self.create_vector((A.T@A).diagonal())

        """ The optimal solution """
        self.work_x = self.create_vector(self.dimN)
        self.data_l = self.create_vector(np_init=l)
        self.data_u = self.create_vector(np_init=u)
        self.data_q = self.create_vector(np_init=q)
        self.pcg_sol = self.create_vector(self.dimN)

    def create_vector(self, size=0, np_init=None):
        if np_init is None:
            return torch.zeros(size, device=self.device, dtype=self.dtype)
        else:
            return torch.from_numpy(np_init, device=self.device, dtype=self.dtype)

    def create_matrix(self, csc_matrix):
        coo_matrix = csc_matrix.tocoo()
        values = torch.tensor(coo_matrix.data, 
                              dtype=torch.float32)
        indices = torch.tensor(
            [coo_matrix.row, coo_matrix.col], 
            dtype=torch.int64)

        sparse_tensor = torch.sparse_coo_tensor(indices, values, coo_matrix.shape)
        return sparse_tensor

    def KKT_mul(self, vec):
        """ multiply the KKT matrix """
        """ first Px0 """
        pcg_P_mul_x = torch.matmul(self.sparse_P, vec)
        """ AtAx0 """
        A_mul_x = torch.matmul(self.sparse_A, vec)
        pcg_At_mul_y = torch.matmul(self.sparse_A.T,
                                            A_mul_x)
        pcg_Kx = pcg_P_mul_x +\
            self.rho*pcg_At_mul_y +\
                self.sigma*vec
        return pcg_Kx

    def solve(self):
        """ osqp_update_rho() in osqp_api.c """
        norm_q = calc_norm_inf(self.data_q)
        rho_recipical = 1.0 / self.rho
        total_pcg_iters = 0

        """ update preconditioner """
        precond = self.diag_PsI + self.rho * self.diag_AtA
        precond = ew_reciprocal(precond)

        """ ----- osqp_solve() in osqp_api.c----- """
        while self.admm_iters < self.max_iter:
            prev_x = self.work_x.clone()
            self.prev_z = work_z.clone()

            """ ----- update_xz_tilde() in osqp_api.c ------ """
            """ compute_rhs() in auxil.c """
            xtilde_view = self.sigma * prev_x - self.data_q
            ztilde_view = rho_recipical * work_y
            ztilde_view = self.prev_z - ztilde_view

            """ PCG preparation """
            """ 0-4 A^T r2 """
            pcg_At_mul_y = torch.matmul(self.sparse_A.T, 
                                        ztilde_view)
            """ 0-5 rhs = r1 + rho*At r2 """
            pcg_computed_rhs = xtilde_view +\
                  self.rho*pcg_At_mul_y; 
            """ KKT multiply x0 """
            pcg_Kx = self.KKT_mul(self.pcg_sol)
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
            pcg_termination = 0
            if norm_pcg_res<pcg_converge_eps or\
                  norm_rhs < self.pcg_eps:
                pcg_termination = 1
            """ PCG loop """
            while pcg_iters < self.pcg_max_iter and\
                  pcg_termination < 1:
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
                prev_r_dot_y = r_dot_y
                """ 37 dot r(k+1) d(k+1) """
                r_dot_y = dot(pcg_res, pcg_d)
                """ 38 compute mu """
                pcg_mu = r_dot_y/prev_r_dot_y
                """ exit if PCG converged """
                if norm_pcg_res<pcg_converge_eps:
                    pcg_termination = 1

                pcg_iters += 1

            xtilde_view = pcg_sol

            total_pcg_iters = total_pcg_iters + pcg_iters
            """ ztilde_view = mat_A * xtilde_view """
            ztilde_view = torch.matmul(self.sparse_A, xtilde_view)
            """ ----- update_x() in osqp_api.c ------ """
            self.work_x = self.alpha * xtilde_view +\
                  self.one_minus_alpha * prev_x
            delta_x = self.work_x - prev_x
            """ ----- update_z() in osqp_api.c ------ """
            alpha_ztilde = self.alpha * ztilde_view + self.one_minus_alpha * self.prev_z
            work_z = alpha_ztilde + rho_recipical * work_y
            work_z = torch.clamp(work_z, self.data_l, self.data_u)
            """ ----- update_y() in osqp_api.c ------ """
            delta_y = alpha_ztilde - work_z
            work_y = work_y + self.rho * delta_y

            """ ------ update_info() in auxil.c ----- """
            if self.admm_iters % self.check_termination == 0:
                """ compute prime residual """
                A_mul_x = torch.matmul(self.sparse_A, self.work_x)
                prev_z = A_mul_x - work_z
                prim_res = calc_norm_inf(prev_z)
                """ compute dual res """
                P_mul_x = torch.matmul(self.sparse_P, self.work_x)
                prev_x = self.data_q + P_mul_x
                At_mul_y = torch.matmul(self.sparse_A.T, work_y)
                prev_x = At_mul_y + prev_x
                dual_res = calc_norm_inf(prev_x)
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
                if prim_res < eps_prim and dual_res < eps_dual:
                    termination = 1
                """ ----- compute_rho_estimate() in auxil.c ----- """
                prim_div = prim_res/prim_norm
                dual_div = dual_res/dual_norm
                rho_estimate = prim_div/dual_div
                rho_estimate = c_sqrt(rho_estimate)
                """ ----- adapt_rho() in osqp_api.c ----- """
                if rho_estimate > 5.0 or rho_estimate < 0.2:
                    rho = rho_estimate 
                    rho_recipical = 1.0/rho
                    """ update preconditioner """
                    precond = self.diag_PsI + self.rho * self.diag_AtA
                    precond = ew_reciprocal(precond)

            self.admm_iters += 1
