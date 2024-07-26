#include "osqp.h"
#include "stdio.h"
#include "xcl2.hpp"

OSQPInt osqp_setup(OSQPSolver**         solverp,
                   const OSQPCscMatrix* P,
                   const OSQPFloat*     q,
                   const OSQPCscMatrix* A,
                   const OSQPFloat*     l,
                   const OSQPFloat*     u,
                   OSQPInt              m,
                   OSQPInt              n,
                   const OSQPSettings*  settings) {
    /* init FPGA */                    
    auto devices = xcl::get_xil_devices();
	return 0;
}

void osqp_set_default_settings(OSQPSettings* settings) {
  /* Avoid working with a null pointer */
    if (!settings){
        return;
    }
	settings->device=0;
}

OSQPInt osqp_solve(OSQPSolver *solver) {
	return 0;
}

OSQPInt osqp_cleanup(OSQPSolver* solver) {
	return 0;
}

OSQPInt osqp_update_data_vec(OSQPSolver*      solver,
                             const OSQPFloat* q_new,
                             const OSQPFloat* l_new,
                             const OSQPFloat* u_new) {
	return 0;							
}

OSQPInt osqp_update_data_mat(OSQPSolver*      solver,
                             const OSQPFloat* Px_new,
                             const OSQPInt*   Px_new_idx,
                             OSQPInt          P_new_n,
                             const OSQPFloat* Ax_new,
                             const OSQPInt*   Ax_new_idx,
                             OSQPInt          A_new_n) {
	return 0;							
}