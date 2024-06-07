void main()
{
	/* ----- osqp_setup() in osqp_api.c----- */
	omegaconf A_multiply, P_multiply, At_multiply;
	// vectors 
	vectorf work_x, prev_x, delta_x, xtilde_view;
	vectorf work_y, delta_y;
	vectorf work_z, prev_z, ztilde_view, ztilde_rhs, alpha_ztilde;
	vectorf data_l, data_u, data_q;
	vectorf A_mul_x, P_mul_x, At_mul_y;
	float const_plus_1 = 1.0;
	float const_zero = 0.0;
	float const_minus_1 = -1.0;
	// solver states
	float admm_iters = 0.0;
	float max_iter = 300.0; //<-
	float prim_res, eps_prim;
	float dual_res, eps_dual;
	float rho = 0.1;
	float rho_recipical;
	float rho_estimate;
	float termination = 0.0;
	float check_termination = 25.0;
	// other scalars
	float sigma = 1e-6;
	float alpha = 1.6;
	float one_minus_alpha = -0.6; // 1 - alpha 
	float settings_eps_abs = 1e-3; 
	float settings_eps_rel = 1e-3;
	float temp_rel_eps;
	float norm_z, norm_Ax, norm_q, norm_Aty, norm_Px;
	float prim_norm, dual_norm;
	float prim_div, dual_div;
	//PCG vectors 
	vectorf diag_PsI, diag_AtA, precond;
	vectorf pcg_computed_rhs;
	vectorf pcg_Kx;
	vectorf pcg_res;
	vectorf pcg_d;
	vectorf pcg_p;
	vectorf pcg_P_mul_x, pcg_At_mul_y;
	vectorf pcg_sol;
	//PCG scalars
	float pcg_max_iter = 20.0;
	float pcg_iters;
	float norm_rhs;
	float pcg_eps=1e-4;
	float pcg_converge_eps;
	float r_dot_y;
	float prev_r_dot_y;
	float norm_pcg_res;
	float pcg_lambda;
	float pcg_mu;
	float p_dot_Kp;
	float pcg_termination = 0.0;
	float total_pcg_iters = 0.0;
	/* DEBUG var*/
	// vectorf test_out;

	/* osqp_update_rho() in osqp_api.c */
	calc_norm_inf(data_q, norm_q);
	rho_recipical = const_plus_1/rho;

	/* update preconditioner */
	precond = diag_PsI + rho*diag_AtA;
    ew_reciprocal(precond, precond);

	/* ----- osqp_solve() in osqp_api.c----- */
	while(admm_iters < max_iter && termination < const_plus_1)
	{
		prev_x = work_x;
		prev_z = work_z;

		/* ----- update_xz_tilde() in osqp_api.c ------ */
		/* compute_rhs() in auxil.c */
		xtilde_view = sigma * prev_x - data_q;
		ztilde_view = rho_recipical * work_y;
		ztilde_view = prev_z + const_minus_1*ztilde_view;

		/* PCG prepare */
		/* 0-4 A^T r2 */
		load_cvb(ztilde_view, any_1, con_1);
		omega_net(At_multiply, any_0);
		cvb_write(pcg_At_mul_y, any_0, sol_1);
		/* 0-5 rhs = r1 + rho*At r2 */
		pcg_computed_rhs = xtilde_view + rho*pcg_At_mul_y; //DEBUG
		/* Px0 */
		cvb_write(pcg_P_mul_x, any_1, sol_1);//clean cvb
		load_cvb(pcg_sol, any_2, sol_1);
		omega_net(P_multiply, any_1);
		cvb_write(pcg_P_mul_x, any_1, sol_1);
		/* AtAx0 */
		omega_net(A_multiply, any_1);
		omega_net(At_multiply, any_0);
		cvb_write(pcg_At_mul_y, any_0, sol_1);

		/* rho is scalar in PCG case */
		pcg_Kx = pcg_P_mul_x + rho * pcg_At_mul_y + sigma * pcg_sol;
		/* 13 r0 = Kx0 - rhs */
		pcg_res = pcg_Kx - pcg_computed_rhs; 
		/* 14. compute d0 = M^-1*r0 */
		ew_prod(pcg_d, precond, pcg_res);

		// test_out = pcg_d; //DEBUG

		/* 15. inf norm of rhs */
		calc_norm_inf(pcg_computed_rhs, norm_rhs);
		/* 16. epsilon times |rhs| */
		pcg_converge_eps = pcg_eps * norm_rhs;
		/* 17. r0 dot d0 */
		dot(r_dot_y, pcg_res, pcg_d);

		/* if pcg_res is 0, skip pcg loop */
		calc_norm_inf(pcg_res, norm_pcg_res);
		/* reset pcg states */
		pcg_mu = const_zero+const_zero;
		pcg_iters = const_zero + const_zero;
		pcg_termination = const_zero + const_zero;
		if (norm_pcg_res<pcg_converge_eps || norm_rhs<pcg_eps)
		{
			pcg_termination = const_plus_1 + const_zero;
		}
		/* PCG loop */
		while(pcg_iters<pcg_max_iter && pcg_termination<const_plus_1)
		{    
			/* 18. p(k+1) = -d(k+1) + mu p(k) */ 
			pcg_p = pcg_mu*pcg_p +const_minus_1*pcg_d;
			/* 19. Kp0 */ 
			cvb_write(pcg_P_mul_x, any_1, sol_1);//clean cvb
			load_cvb(pcg_p, any_2, sol_1);
			omega_net(P_multiply, any_1);
			cvb_write(pcg_P_mul_x, any_1, sol_1);
			omega_net(A_multiply, any_1);
			omega_net(At_multiply, any_0);
			cvb_write(pcg_At_mul_y, any_0, sol_1);
			pcg_Kx = sigma * pcg_p + pcg_P_mul_x + rho*pcg_At_mul_y;
			/* 30. p dot Kp0*/
			dot(p_dot_Kp, pcg_p, pcg_Kx);
			// 31. lambda = r_dot_y/p_dot_Kp */
			pcg_lambda = r_dot_y/p_dot_Kp;
			// 32 x(k+1)=x(k)+lambda p(k)*/
			pcg_sol = pcg_sol + pcg_lambda * pcg_p;
			// 33 r(k+1)=x(k)+lambda Kp*/
			pcg_res = pcg_res + pcg_lambda*pcg_Kx;
			// 34 |res| 
			calc_norm_inf(pcg_res, norm_pcg_res);
			// 35 d = M^-1 r 
			ew_prod(pcg_d, precond, pcg_res);
			// 36 save r_dot_y 
			prev_r_dot_y = r_dot_y+const_zero;
			// 37 dot r(k+1) d(k+1) 
			dot(r_dot_y, pcg_res , pcg_d);
			// 38 compute mu 
			pcg_mu = r_dot_y/prev_r_dot_y;
			// exit if PCG converged 
			if (norm_pcg_res<pcg_converge_eps)
			{
				pcg_termination = const_plus_1 + const_zero;
			}

			pcg_iters = pcg_iters + const_plus_1;
		}
		xtilde_view = pcg_sol;

		total_pcg_iters = total_pcg_iters + pcg_iters;

		/* ztilde_view = mat_A * xtilde_view*/
		load_cvb(xtilde_view, any_1, sol_1);
		omega_net(A_multiply, any_0);
		cvb_write(ztilde_view, any_0, con_1);
		// ew_prod(ztilde_view, rho_inv_vec, ztilde_view); 
		// ztilde_view = ztilde_view + ztilde_rhs;

		/* ----- update_x() in osqp_api.c ------ */
		work_x = alpha * xtilde_view + one_minus_alpha * prev_x;
		delta_x = work_x - prev_x;
		/* ----- update_z() in osqp_api.c ------ */
		alpha_ztilde = alpha * ztilde_view + one_minus_alpha * prev_z;
		work_z = alpha_ztilde + rho_recipical*work_y;
		work_z = work_z > data_l;
		work_z = work_z < data_u;
		/* ----- update_y() in osqp_api.c ------ */
		delta_y = alpha_ztilde - work_z;
		work_y = work_y + rho*delta_y;

		/* ------update_info() in auxil.c -----*/
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

			/* ----- compute_rho_estimate() in auxil.c ----- */
			prim_div = prim_res/prim_norm;
			dual_div = dual_res/dual_norm;
			rho_estimate = prim_div/dual_div;
			c_sqrt(rho_estimate, rho_estimate);

			/* ----- adapt_rho() in osqp_api.c ----- */
			if (rho_estimate > 5.0 || rho_estimate < 0.2){
				rho_estimate = rho * rho_estimate;
				rho = const_plus_1 * rho_estimate;
				rho_recipical = const_plus_1/rho;
				/* update preconditioner */
				precond = diag_PsI + rho*diag_AtA;
    			ew_reciprocal(precond, precond);
			}
		}
		admm_iters = admm_iters + const_plus_1;
	}
}