void main()
{
	omegaconf LowerSolve, DiagSolve, UpperSolve, Perm, Perm_inv;
	vectorf in_x, in_z, out_x, out_z;

	load_cvb(in_x, sol_1_con_1, sol_1);
	load_cvb(in_z, sol_2_con_1, con_1);

	omega_net(Perm, any_0);
	omega_net(LowerSolve, any_0);
	omega_net(DiagSolve, any_0);
	omega_net(UpperSolve, any_0);
	omega_net(Perm_inv, any_0);

	cvb_write(out_x, sol_1_con_1, sol_1);
	cvb_write(out_z, sol_2_con_1, con_1);

}