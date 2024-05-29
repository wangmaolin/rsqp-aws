void main()
{
	omegaconf SpMat;
	vectorf spmv_in, test_out;

	// clear partial sum results
	vectorf zero_vec;
	load_cvb(zero_vec, any_2, any_1);

	load_cvb(spmv_in, any_1, con_1);
	omega_net(SpMat, any_0);
	cvb_write(test_out, any_0, sol_1);
}