#ifndef OSQP_API_TYPES_H
#define OSQP_API_TYPES_H

#include <vector>
#include "xcl2.hpp" // FPGA runtime header 

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
  OSQPInt elf[16];
  OSQPInt hbmTotalChannels;
  OSQPInt lr_mem_pc_words;
  OSQPInt nnz_mem_pc_words;
} OSQPInfo;

typedef std::vector<float, aligned_allocator<float>> align_floats; 
typedef std::vector<unsigned int, aligned_allocator<unsigned int>> align_uints;

typedef struct {
  OSQPSettings*  settings; ///< Problem settings
  OSQPSolution*  solution; ///< Computed solution
  OSQPInfo*      info;     ///< Solver information

  std::vector<align_floats> host_matrix; // matrix workspace on the CPU side
	std::vector<cl::Buffer> hbm_matrix; // matrix workspace on the FPGA side
  std::vector<align_floats> host_vec;
	std::vector<cl::Buffer> hbm_vec;
  cl::CommandQueue cmd_queue;
  cl::Kernel cu_krnl;
} OSQPSolver;

#endif /* ifndef OSQP_API_TYPES_H */