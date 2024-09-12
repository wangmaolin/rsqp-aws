#include "osqp.h"
#include "stdio.h"

#include "xcl2.hpp" // FPGA runtime header 
#include <cstring>
#include <iomanip>
#include <vector>
#include <iostream>
#include <Python.h>

#define DATA_PACK_NUM 16
void save_results(float * memPtr, int memLen){
    std::ofstream vector_file("./temp/result_vector.txt");
    for(int loc=0; loc<memLen; loc++){
        vector_file<<memPtr[loc]<<"\n";
    }
    vector_file.close();
}

int test_py_call() {
    // Initialize the Python interpreter
    Py_Initialize();

    // Import the Python script
    PyObject* pName = PyUnicode_DecodeFSDefault("my_python_script");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != nullptr) {
        // Get the function from the module
        PyObject* pFunc = PyObject_GetAttrString(pModule, "process_array");

        if (pFunc && PyCallable_Check(pFunc)) {
            // Create a Python list from a C++ array
            std::vector<int> cpp_array = {1, 2, 3, 4, 5};
            PyObject* pList = PyList_New(cpp_array.size());
            for (size_t i = 0; i < cpp_array.size(); ++i) {
                PyList_SetItem(pList, i, PyLong_FromLong(cpp_array[i]));
            }

            // Call the Python function with the list
            PyObject* pArgs = PyTuple_Pack(1, pList);
            PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pValue != nullptr) {
                // Convert the result to a C++ type and print it
                long result = PyLong_AsLong(pValue);
                std::cout << "Result from Python: " << result << std::endl;
                Py_DECREF(pValue);
            } else {
                PyErr_Print();
                std::cerr << "Call to process_array failed" << std::endl;
            }

            Py_DECREF(pList);
            Py_DECREF(pFunc);
        } else {
            PyErr_Print();
            std::cerr << "Cannot find function 'process_array'" << std::endl;
        }

        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        std::cerr << "Failed to load 'my_python_script'" << std::endl;
    }

    // Finalize the Python interpreter
    Py_Finalize();

    return 0;
}

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
    OSQPSolver* solver; 
    solver = new OSQPSolver;
    // calloc(1, sizeof(OSQPSolver));
    *solverp = solver;
    // Allocate empty info struct
    solver->info = new OSQPInfo;

    // read meta infos about the program 
    std::ifstream inst_file_stream(settings->elf, std::ifstream::binary);
    inst_file_stream.read(reinterpret_cast<char*>(solver->info->elf), 
        sizeof(solver->info->elf));

	solver->info->hbmTotalChannels = solver->info->elf[12];
	solver->info->nnz_mem_pc_words = solver->info->elf[14];
	solver->info->lr_mem_pc_words = solver->info->elf[15];

    if (solver->info->elf[0]!=2135247942){ // check header signature 
        std::cout << "WRONG BIN FILE!:" << std::endl;
        return EXIT_FAILURE;
    }

	cl_int hw_err;
    cl::Context context;
    cl::Program program;
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
    OCL_CHECK(hw_err, solver->cmd_queue = cl::CommandQueue(context, device, cl::QueueProperties::OutOfOrder, &hw_err));
    program = cl::Program(context, {device}, bins, nullptr, &hw_err);

    // python funcion call test
    test_py_call();

    // ------ Setup Profile End ------ 
    auto setup_end= std::chrono::high_resolution_clock::now();
    setup_time = std::chrono::duration<double>(setup_end - setup_start);
    std::cout << "FPGA setup time:  "<<
    std::setprecision(2) << std::scientific
    << setup_time.count()<<"s"<<std::endl;

    // Instruction Transfer Profile Start 
    std::chrono::duration<double> dt_time(0);
    auto dt_start = std::chrono::high_resolution_clock::now();

    std::string cu_id = std::to_string(1);
    std::string cu_krnls_name_full = std::string("cu_top")+ std::string(":{") + std::string("cu_top_") + cu_id + std::string("}");
    OCL_CHECK(hw_err, solver->cu_krnl = cl::Kernel(program, cu_krnls_name_full.c_str(), &hw_err));

    // transfer compiled instructions 
    int indice_mem_words = solver->info->elf[5] * DATA_PACK_NUM ;
    std::vector<unsigned int, aligned_allocator<unsigned int>> host_indice_buf(indice_mem_words);
    inst_file_stream.read(reinterpret_cast<char *>(host_indice_buf.data()), indice_mem_words * sizeof(unsigned int));
    cl::Buffer cu_indice_mem;
    int krnlArgCount=0;
    OCL_CHECK(hw_err, cu_indice_mem = cl::Buffer(context,
                                                 CL_MEM_USE_HOST_PTR,
                                                 indice_mem_words * sizeof(unsigned int),
                                                 host_indice_buf.data(),
                                                 &hw_err));
    OCL_CHECK(hw_err, hw_err = solver->cu_krnl.setArg(krnlArgCount++, cu_indice_mem));
    solver->cmd_queue.enqueueMigrateMemObjects({cu_indice_mem}, 0);// 0 means from host
    solver->cmd_queue.finish();

    // Instruction Transfer End 
    auto dt_end = std::chrono::high_resolution_clock::now();
    dt_time = std::chrono::duration<double>(dt_end - dt_start);
    std::cout << "Instruction Transfer time:  "<<
    std::setprecision(2) << std::scientific
    << dt_time.count()<<"s"<<std::endl;

    // allocate Vector Region on HBM
	for(int hbmItem=0; hbmItem<solver->info->hbmTotalChannels; hbmItem++) {
        solver->host_vec.push_back(align_floats(solver->info->lr_mem_pc_words));

        inst_file_stream.read(reinterpret_cast<char *>(solver->host_vec[hbmItem].data()),
            solver->info->lr_mem_pc_words * sizeof(float));

        cl::Buffer bufItem; 
        OCL_CHECK(hw_err, bufItem=cl::Buffer(context,
                                              CL_MEM_USE_HOST_PTR,
                                              solver->info->lr_mem_pc_words * sizeof(float),
                                              solver->host_vec[hbmItem].data(),
                                              &hw_err));
        solver->hbm_vec.push_back(bufItem);

        OCL_CHECK(hw_err, hw_err = solver->cu_krnl.setArg(krnlArgCount++,
                                                  solver->hbm_vec[hbmItem]));

    }
    OCL_CHECK(hw_err, hw_err = solver->cu_krnl.setArg(krnlArgCount++,
                                              solver->info->elf[1]));
    OCL_CHECK(hw_err, hw_err = solver->cu_krnl.setArg(krnlArgCount++,
                                              0));
	return 0;
}

void osqp_set_default_settings(OSQPSettings* settings) {
    if (!settings){ // Avoid working with a null pointer 
        return;}

	// settings->device=1;
    // strcpy(settings->xclbin, "./temp/u50-2-nk1-4000.xclbin");

	settings->device=0;
    strcpy(settings->xclbin, "./temp/u50-2-Jul25-4000.xclbin");

    strcpy(settings->elf, "./temp/test.fpga");
}

OSQPInt osqp_solve(OSQPSolver *solver) {
    //Profiling
    double kernel_time_in_sec = 0;
    std::chrono::duration<double> kernel_time(0);
    auto kernel_start = std::chrono::high_resolution_clock::now();
    solver->cmd_queue.enqueueTask(solver->cu_krnl);
    solver->cmd_queue.finish();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
    kernel_time_in_sec = kernel_time.count();
    // run time in second 
    std::cout << "---- Solver Run time:  "<<
    std::setprecision(2) << std::scientific
    << kernel_time_in_sec<<"s"<<std::endl;

    // Copy register file back 
    solver->cmd_queue.enqueueMigrateMemObjects({solver->hbm_vec[0]},
                                       CL_MIGRATE_MEM_OBJECT_HOST);
    solver->cmd_queue.finish();

    std::ofstream register_file("./temp/reg_content.txt");
    for (int i=0; i<64; i++){
        register_file<<solver->host_vec[0].data()[i]<<"\n";
    }
    register_file.close();

    // Copy solution back, combining multiple HBM channels 
    int iscaC = DATA_PACK_NUM*solver->info->hbmTotalChannels;
    int channelAddr = (solver->info->elf[9] >>1)*DATA_PACK_NUM; 
    int channelPacks = solver->info->elf[10];
    std::vector<float> combinedResult(channelPacks*iscaC);
    
	for(int hbmItem=0; hbmItem<solver->info->hbmTotalChannels; hbmItem++) {
        // Transfer HBM channel 
        solver->cmd_queue.enqueueMigrateMemObjects({solver->hbm_vec[hbmItem]},
                                        CL_MIGRATE_MEM_OBJECT_HOST);
		solver->cmd_queue.finish();
        // Interleave Results 
        for(int pItem=0; pItem<channelPacks; pItem++){
            for(int dItem=0; dItem<DATA_PACK_NUM; dItem++){
                int srcAddr = channelAddr + pItem*DATA_PACK_NUM+dItem;
                int dstAddr = pItem*iscaC + hbmItem*DATA_PACK_NUM + dItem;
                combinedResult.data()[dstAddr] = solver->host_vec[hbmItem].data()[srcAddr];
            }
        }
    }
    save_results(combinedResult.data(), channelPacks*iscaC);

	return 0;
}

OSQPInt osqp_update_data_vec(OSQPSolver*      solver,
                             const OSQPFloat* q_new,
                             const OSQPFloat* l_new,
                             const OSQPFloat* u_new) {
    // ----- Vector Transfer Profile Start -----
    std::chrono::duration<double> dt_time(0);
    auto dt_start = std::chrono::high_resolution_clock::now();

	for(int i=0; i<solver->info->hbmTotalChannels;i++) {
        solver->cmd_queue.enqueueMigrateMemObjects({solver->hbm_vec[i]}, 0 );
		solver->cmd_queue.finish();
	}

    // ---- Vector Transfer End  -----
    auto dt_end = std::chrono::high_resolution_clock::now();
    dt_time = std::chrono::duration<double>(dt_end - dt_start);
    std::cout << "Vector Transfer time:  "<<
    std::setprecision(2) << std::scientific
    << dt_time.count()<<"s"<<std::endl;
	return 0;							
}

OSQPInt osqp_update_data_mat(OSQPSolver*      solver,
                             const OSQPFloat* Px_new,
                             const OSQPInt*   Px_new_idx,
                             OSQPInt          P_new_n,
                             const OSQPFloat* Ax_new,
                             const OSQPInt*   Ax_new_idx,
                             OSQPInt          A_new_n) {
    /* the matrix data also stores at hbm_vec */
	return 0;							
}

OSQPInt osqp_cleanup(OSQPSolver* solver) {
    // Free information
    delete solver->info;

    // Free solver
    delete solver;
	return 0;
}
