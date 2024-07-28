#ifndef OSQP_API_TYPES_H
#define OSQP_API_TYPES_H

typedef int OSQPInt;       /* for indices */
typedef float OSQPFloat;  /* for numerical values  */

typedef struct {
  OSQPInt    m;     ///< number of rows
  OSQPInt    n;     ///< number of columns
  OSQPInt   *p;     ///< column pointers (size n+1); col indices (size nzmax) starting from 0 for triplet format
  OSQPInt   *i;     ///< row indices, size nzmax starting from 0
  OSQPFloat *x;     ///< numerical values, size nzmax
  OSQPInt    nzmax; ///< maximum number of entries
  OSQPInt    nz;    ///< number of entries in triplet matrix, -1 for csc
} OSQPCscMatrix;

typedef struct {
  char    elf[32];     ///< instructions
  char    xclbin[32];     ///< bitstream
  OSQPInt device; ///< FPGA identifier
} OSQPSettings;

typedef struct {
  OSQPFloat* x;             ///< Primal solution
  OSQPFloat* y;             ///< Lagrange multiplier associated with \f$l \le Ax \le u\f$
  OSQPFloat* prim_inf_cert; ///< Primal infeasibility certificate
  OSQPFloat* dual_inf_cert; ///< Dual infeasibility certificate
} OSQPSolution;

typedef struct {
  // solver status
  char    status[32];     ///< Status string, e.g. 'solved'
} OSQPInfo;

typedef struct {
  OSQPSettings*  settings; ///< Problem settings
  OSQPSolution*  solution; ///< Computed solution
  OSQPInfo*      info;     ///< Solver information
} OSQPSolver;

#endif /* ifndef OSQP_API_TYPES_H */