#ifndef  OSQP_API_FUNCTIONS_H
#define  OSQP_API_FUNCTIONS_H

#include "osqp_api_types.h"

void osqp_set_default_settings(OSQPSettings* settings);

OSQPInt osqp_setup(OSQPSolver**         solverp,
                   const OSQPCscMatrix* P,
                   const OSQPFloat*     q,
                   const OSQPCscMatrix* A,
                   const OSQPFloat*     l,
                   const OSQPFloat*     u,
                   OSQPInt              m,
                   OSQPInt              n,
                   const OSQPSettings*  settings); 

OSQPInt osqp_solve(OSQPSolver* solver);

OSQPInt osqp_cleanup(OSQPSolver* solver); 

OSQPInt osqp_update_data_vec(OSQPSolver*      solver,
                             const OSQPFloat* q_new,
                             const OSQPFloat* l_new,
                             const OSQPFloat* u_new); 

OSQPInt osqp_update_data_mat(OSQPSolver*      solver,
                             const OSQPFloat* Px_new,
                             const OSQPInt*   Px_new_idx,
                             OSQPInt          P_new_n,
                             const OSQPFloat* Ax_new,
                             const OSQPInt*   Ax_new_idx,
                             OSQPInt          A_new_n);

#endif /* ifndef OSQP_API_FUNCTIONS_H */
