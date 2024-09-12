void main()
{
	float const_plus_1 = 1.0;
	float mul_a = 2.0;
	float mul_b = -1.0;
	float mul_c=1.5;
	float ut_scalar;
	omegaconf vector_sum; //required by dot

	vectorf vec_a, vec_b, vec_c; 
	vectorf test_out;

	test_out = mul_a*vec_a + mul_b*vec_b + mul_c*vec_c;
	// ew_prod(test_out, vec_a, vec_b);
	
	// vectorf zero_vec;
	// load_cvb(zero_vec, any_0, any_1);
	dot(ut_scalar, vec_a, vec_b);
}