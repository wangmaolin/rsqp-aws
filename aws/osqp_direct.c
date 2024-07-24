void main()
{
	// ----- osqp_setup() in osqp_api.c----- 
	// ldl solve 
	omegaconf LowerSolve, DiagSolve, UpperSolve, Perm, Perm_inv;
	// spmv
	omegaconf  A_multiply, P_multiply, At_multiply;
	// vectors 
	vectorf work_x, prev_x, delta_x, xtilde_view;
	vectorf work_y, delta_y;
	vectorf work_z, prev_z, ztilde_view, ztilde_rhs, alpha_ztilde;
	vectorf data_l, data_u, data_q;
	vectorf rho_vec, rho_inv_vec, constr_type;
	vectorf A_mul_x, P_mul_x, At_mul_y;
	float const_plus_1 = 1.0;
	// solver states
	float admm_iters = 0.0;
	float max_iter = 300.0; //<-
	float prim_res, eps_prim;
	float dual_res, eps_dual;
	float rho = 0.1;
	float rho_estimate;
	float termination = 0.0;
	float check_termination = 25.0;
	// other scalars
	float const_zero = 0.0;
	float const_minus_1 = -1.0;
	float sigma = 1e-6;
	float alpha = 1.6;
	float one_minus_alpha = -0.6; // 1 - alpha 
	float OSQP_RHO_MIN = 1e-6;
	float rho_eq_constr;
	float OSQP_RHO_EQ_OVER_RHO_INEQ = 1e3;
	float settings_eps_abs = 1e-3; 
	float settings_eps_rel = 1e-3;
	float temp_rel_eps;
	float norm_z, norm_Ax, norm_q, norm_Aty, norm_Px;
	float prim_norm, dual_norm;
	float prim_div, dual_div;
	float total_rho_update=0.0;

	// osqp_update_rho() in osqp_api.c 
	calc_norm_inf(data_q, norm_q);
	rho_eq_constr = OSQP_RHO_EQ_OVER_RHO_INEQ * rho;

	set_scalar_conditional(rho_vec, constr_type, OSQP_RHO_MIN, rho, rho_eq_constr);
    ew_reciprocal(rho_inv_vec, rho_vec);

	// ----- osqp_solve() in osqp_api.c ----- 
	while(admm_iters < max_iter && termination < const_plus_1)
	{
		prev_x = work_x;
		prev_z = work_z;
		// ----- update_xz_tilde() in osqp_api.c ------ 
		// compute_rhs() in auxil.c 
		xtilde_view = sigma * prev_x - data_q;
		ew_prod(ztilde_view, rho_inv_vec , work_y); 
		ztilde_view = prev_z - ztilde_view;
		ztilde_rhs = ztilde_view;

		// linsys_solver->solve() in osqp_api.c 
		load_cvb(xtilde_view, sol_1_con_1, sol_1);
		load_cvb(ztilde_view, sol_2_con_1, con_1);

		omega_net(Perm, any_0);
		omega_net(LowerSolve, any_0);
		omega_net(DiagSolve, any_0);
		omega_net(UpperSolve, any_0);
		omega_net(Perm_inv, any_0);

		cvb_write(xtilde_view, sol_1_con_1, sol_1);
		cvb_write(ztilde_view, sol_2_con_1, con_1);

		ew_prod(ztilde_view, rho_inv_vec, ztilde_view); 
		ztilde_view = ztilde_view + ztilde_rhs;

		// ----- update_x() in osqp_api.c ------ 
		work_x = alpha * xtilde_view + one_minus_alpha * prev_x;
		delta_x = work_x - prev_x;

		// ----- update_z() in osqp_api.c ------ 
    	ew_prod(work_z, rho_inv_vec, work_y);
		alpha_ztilde = alpha * ztilde_view + one_minus_alpha * prev_z;
		work_z = alpha_ztilde + work_z;
		work_z = work_z > data_l;
		work_z = work_z < data_u;

		// ----- update_y() in osqp_api.c ------ 
		delta_y = alpha_ztilde - work_z;
		ew_prod(delta_y, rho_vec, delta_y);
		work_y = work_y + delta_y;

		// ------update_info() in auxil.c -----
		if ((admm_iters % check_termination)<const_plus_1){
			// compute prime residual
			load_cvb(work_x, any_1, sol_1); 
			omega_net(A_multiply, any_0);
			cvb_write(A_mul_x, any_0, con_1);
			prev_z = A_mul_x - work_z;
			calc_norm_inf(prev_z, prim_res);	

			// compute_dual_res 
			cvb_write(P_mul_x, any_0, sol_1);//clean cvb
			omega_net(P_multiply, any_0);
			cvb_write(P_mul_x, any_0, sol_1);
			prev_x = data_q + P_mul_x;
			load_cvb(work_y, any_1, con_1); 
			omega_net(At_multiply, any_0);
			cvb_write(At_mul_y, any_0, sol_1);
			prev_x = At_mul_y + prev_x;
			calc_norm_inf(prev_x, dual_res);

			// compute_prim_tol 
			calc_norm_inf(work_z, norm_z);
			calc_norm_inf(A_mul_x, norm_Ax);
			select_max(prim_norm, norm_z, norm_Ax);
			eps_prim = settings_eps_rel * prim_norm;
			eps_prim = settings_eps_abs + eps_prim;

			// compute_dual_tol 
			calc_norm_inf(At_mul_y, norm_Aty);
			select_max(temp_rel_eps, norm_q, norm_Aty);
			calc_norm_inf(P_mul_x, norm_Px);
			select_max(dual_norm, temp_rel_eps, norm_Px);
			eps_dual = settings_eps_rel * dual_norm;
			eps_dual = settings_eps_abs + eps_dual;

			// termination check 
			if((prim_res < eps_prim) && (dual_res < eps_dual)){
				termination = termination + const_plus_1;
			}

			// ----- compute_rho_estimate() in auxil.c ----- 
			prim_div = prim_res/prim_norm;
			dual_div = dual_res/dual_norm;
			rho_estimate = prim_div/dual_div;
			c_sqrt(rho_estimate, rho_estimate);

			// ----- adapt_rho() in osqp_api.c ----- 
			if (rho_estimate > 5.0 || rho_estimate < 0.2){
				rho_estimate = rho * rho_estimate;
				rho = const_plus_1 * rho_estimate;
				set_scalar_conditional(rho_vec, constr_type, OSQP_RHO_MIN, rho, rho_eq_constr);
				ew_reciprocal(rho_inv_vec, rho_vec);
				total_rho_update = total_rho_update + const_plus_1;
				// break at the first factor 
				termination = termination + const_plus_1;
			}
		}
		admm_iters = admm_iters + const_plus_1;
	}
}