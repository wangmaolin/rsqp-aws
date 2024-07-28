#include "osqp.h"
#include "stdio.h"

#include "xcl2.hpp" // FPGA runtime header 
#include <cstring>
#include <iomanip>

#define DATA_PACK_NUM 16

OSQPInt osqp_setup(OSQPSolver**         solverp,
                   const OSQPCscMatrix* P,
                   const OSQPFloat*     q,
                   const OSQPCscMatrix* A,
                   const OSQPFloat*     l,
                   const OSQPFloat*     u,
                   OSQPInt              m,
                   OSQPInt              n,
                   const OSQPSettings*  settings) {
    // Allocate empty solver
    solver = c_calloc(1, sizeof(OSQPSolver));
    if (!(solver)) return osqp_error(OSQP_MEM_ALLOC_ERROR);
    *solverp = solver;

    // read accelerator program 
    std::ifstream inst_file_stream(settings->elf, std::ifstream::binary);

    int meta_info_size = 16;
    std::vector<uint32_t> elf_info(meta_info_size);
    // read meta infos about the program 
    inst_file_stream.read(reinterpret_cast<char*>(elf_info.data()), 4*meta_info_size);
    if (elf_info[0]!=2135247942){ // check header signature 
        std::cout << "WRONG BIN FILE!:" << std::endl;
        return EXIT_FAILURE;
    }

	cl_int hw_err;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue cmd_queue;
    cl::Kernel cu_krnl;

    //------ Setup Profile Start ------ 
    std::chrono::duration<double> setup_time(0);
    auto setup_start = std::chrono::high_resolution_clock::now();

    auto devices = xcl::get_xil_devices();
    std::string binaryFile(settings->xclbin);
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    // Select FPGA ID
    auto device = devices[settings->device];
    OCL_CHECK(hw_err, context = cl::Context(device, nullptr, nullptr, nullptr, &hw_err));
    OCL_CHECK(hw_err, cmd_queue = cl::CommandQueue(context, device, cl::QueueProperties::OutOfOrder, &hw_err));
    program = cl::Program(context, {device}, bins, nullptr, &hw_err);

    // ------ Setup Profile End ------ 
    auto setup_end= std::chrono::high_resolution_clock::now();
    setup_time = std::chrono::duration<double>(setup_end - setup_start);
    std::cout << "FPGA setup time:  "<<
    std::setprecision(2) << std::scientific
    << setup_time.count()<<"s"<<std::endl;

    // Data Transfer Profile Start 
    std::chrono::duration<double> dt_time(0);
    auto dt_start = std::chrono::high_resolution_clock::now();

    std::string cu_id = std::to_string(1);
    std::string cu_krnls_name_full = std::string("cu_top")+ std::string(":{") + std::string("cu_top_") + cu_id + std::string("}");
    OCL_CHECK(hw_err, cu_krnl = cl::Kernel(program, cu_krnls_name_full.c_str(), &hw_err));

    // transfer compiled instructions 
    int indice_mem_words = elf_info[5] * DATA_PACK_NUM ;
    std::vector<unsigned int, aligned_allocator<unsigned int>> host_indice_buf(indice_mem_words);
    inst_file_stream.read(reinterpret_cast<char *>(host_indice_buf.data()), indice_mem_words * sizeof(unsigned int));
    cl::Buffer cu_indice_mem;
    int krnlArgCount=0;
    OCL_CHECK(hw_err, cu_indice_mem = cl::Buffer(context,
                                                 CL_MEM_USE_HOST_PTR,
                                                 indice_mem_words * sizeof(unsigned int),
                                                 host_indice_buf.data(),
                                                 &hw_err));
    OCL_CHECK(hw_err, hw_err = cu_krnl.setArg(krnlArgCount++, cu_indice_mem));
    cmd_queue.enqueueMigrateMemObjects({cu_indice_mem}, 0);// 0 means from host
    cmd_queue.finish();

    // Data Transfer End 
    auto dt_end = std::chrono::high_resolution_clock::now();
    dt_time = std::chrono::duration<double>(dt_end - dt_start);
    std::cout << "Instruction Transfer time:  "<<
    std::setprecision(2) << std::scientific
    << dt_time.count()<<"s"<<std::endl;

	return 0;
}

void osqp_set_default_settings(OSQPSettings* settings) {
    if (!settings){ // Avoid working with a null pointer 
        return;}
	settings->device=1;
    strcpy(settings->xclbin, "./temp/u50-2-nk1-4000.xclbin");
    strcpy(settings->elf, "./temp/test.fpga");
}

OSQPInt osqp_solve(OSQPSolver *solver) {
    /*
    //Profiling
    double kernel_time_in_sec = 0;
    std::chrono::duration<double> kernel_time(0);
    auto kernel_start = std::chrono::high_resolution_clock::now();
    cmd_queue.enqueueTask(cu_krnl);
    cmd_queue.finish();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
    kernel_time_in_sec = kernel_time.count();
    // run time in second 
    std::cout << "FPGA run time:  "<<
    std::setprecision(2) << std::scientific
    << kernel_time_in_sec<<"s"<<std::endl;

    // Copy register file back 
    cmd_queue.enqueueMigrateMemObjects({cu_rhs_mem[0]},
                                       CL_MIGRATE_MEM_OBJECT_HOST);
    cmd_queue.finish();

    std::ofstream register_file("./temp/reg_content.txt");
    for (int i=0; i<64; i++){
        register_file<<host_lr_buf.data()[i]<<"\n";
    }
    register_file.close();

    // Copy solution back, combining multiple HBM channels 
    int iscaC = DATA_PACK_NUM*hbmTotalChannels;
    int channelAddr = (elf_info[9] >>1)*DATA_PACK_NUM; 
    int channelPacks = elf_info[10];
    std::vector<float> combinedResult(channelPacks*iscaC);
    
	for(int hbmItem=0; hbmItem<hbmTotalChannels; hbmItem++) {
        // Transfer HBM channel 
        cmd_queue.enqueueMigrateMemObjects({cu_rhs_mem[hbmItem]},
                                        CL_MIGRATE_MEM_OBJECT_HOST);
		cmd_queue.finish();
        // Interleave Results 
        for(int pItem=0; pItem<channelPacks; pItem++){
            for(int dItem=0; dItem<DATA_PACK_NUM; dItem++){
                int srcAddr = channelAddr + pItem*DATA_PACK_NUM+dItem;
                int dstAddr = pItem*iscaC + hbmItem*DATA_PACK_NUM + dItem;
                combinedResult.data()[dstAddr] = host_lr_buf.data()[srcAddr];
            }
        }
    }
    save_results(combinedResult.data(), channelPacks*iscaC);
    */ 

	return 0;
}

OSQPInt osqp_update_data_vec(OSQPSolver*      solver,
                             const OSQPFloat* q_new,
                             const OSQPFloat* l_new,
                             const OSQPFloat* u_new) {
    // transfer vector                               
	int lr_mem_pc_words = elf_info[15];
	std::vector<float, aligned_allocator<float>> host_lr_buf(lr_mem_pc_words);

	std::vector<cl::Buffer> cu_rhs_mem(hbmTotalChannels);

    /*
	for(int i=0; i<hbmTotalChannels;i++) 
	{
        inst_file_stream.read(reinterpret_cast<char *>(host_lr_buf.data()),
                              lr_mem_pc_words * sizeof(float));
        OCL_CHECK(hw_err, cu_rhs_mem[i] = cl::Buffer(context,
                                              CL_MEM_USE_HOST_PTR,
                                              lr_mem_pc_words * sizeof(float),
                                              host_lr_buf.data(),
                                              &hw_err));
        OCL_CHECK(hw_err, hw_err = cu_krnl.setArg(krnlArgCount++,
                                                  cu_rhs_mem[i]));
        cmd_queue.enqueueMigrateMemObjects({cu_rhs_mem[i]}, 0 );
		cmd_queue.finish();
	}

    OCL_CHECK(hw_err, hw_err = cu_krnl.setArg(krnlArgCount++,
                                              elf_info[1]));
    OCL_CHECK(hw_err, hw_err = cu_krnl.setArg(krnlArgCount++,
                                              0));
    */
	return 0;							
}

OSQPInt osqp_update_data_mat(OSQPSolver*      solver,
                             const OSQPFloat* Px_new,
                             const OSQPInt*   Px_new_idx,
                             OSQPInt          P_new_n,
                             const OSQPFloat* Ax_new,
                             const OSQPInt*   Ax_new_idx,
                             OSQPInt          A_new_n) {
    /*
    // transfer matrix data 
	int hbmTotalChannels = elf_info[12];
	int nnz_mem_pc_words = elf_info[14];
	std::vector<cl::Buffer>  cu_nnz_mem(hbmTotalChannels);
	std::vector<float, aligned_allocator<float>> host_nnz_buf(nnz_mem_pc_words);
	for(int i=0; i<hbmTotalChannels;i++) 
	{
		inst_file_stream.read(reinterpret_cast<char *>(host_nnz_buf.data()), nnz_mem_pc_words*sizeof(float));
		OCL_CHECK(hw_err, cu_nnz_mem[i] = cl::Buffer(context,
													CL_MEM_USE_HOST_PTR, 
													nnz_mem_pc_words* sizeof(float),
													host_nnz_buf.data(), 
													&hw_err));
		OCL_CHECK(hw_err, hw_err = cu_krnl.setArg(krnlArgCount++, cu_nnz_mem[i]));
		cmd_queue.enqueueMigrateMemObjects({cu_nnz_mem[i]}, 0 );
		cmd_queue.finish();
	}
    */
	return 0;							
}

OSQPInt osqp_cleanup(OSQPSolver* solver) {
    // Free solver
    c_free(solver);
	return 0;
}
