void main()
{
	float const_plus_1 = 1.0;
	float mul_a = 1.0;
	float mul_b = -2.0;
	float mul_c=2.0;
	vectorf vec_a, vec_b, vec_c; 
	// vectorf zero_vec;

	vectorf test_out;
	float test_scalar;

	// test_out = mul_a*vec_a + mul_b*vec_b;
	test_out = mul_a*vec_a + mul_b*vec_b + mul_c*vec_c;
	// ew_prod(test_out, vec_a, vec_b);
	
	// load_cvb(zero_vec, any_0, any_1);
	dot(test_scalar, vec_a, vec_b);
}