void main()
{
	omegaconf LowerSolve; 
	// omegaconf DiagSolve, UpperSolve;

	vectorf in_x; 
	vectorf in_z; 
	vectorf out_x;
	vectorf out_z;

	load_cvb(in_x, any_0, sol_1);
	load_cvb(in_z, sol_1, con_1);
	omega_net(LowerSolve, any_0);
	// omega_net(DiagSolve, any_0);
	// omega_net(UpperSolve, any_0);
	cvb_write(out_x, any_0, sol_1);
	cvb_write(out_z, sol_1, con_1);

}