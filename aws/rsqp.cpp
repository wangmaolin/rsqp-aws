#include <algorithm>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include "xcl2.hpp"
#include "cmdlineparser.h"

#define DATA_PACK_NUM 16

void save_results(float * memPtr, int memLen){
    std::ofstream vector_file("./temp/result_vector.txt");
    for(int loc=0; loc<memLen; loc++){
        vector_file<<memPtr[loc]<<"\n";
    }
    vector_file.close();
}

int main(int argc, char** argv) {
    /* Command Line Parser <Full Arg>, <Short Arg>, <Description>, <Default> */
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "./proc.xclbin");
    parser.addSwitch("--program_name", "-p", "accelerator programm", "./cosim.fpga");
    parser.parse(argc, argv);

    /* read accelerator program */
    std::ifstream inst_file_stream(parser.value("program_name"), std::ifstream::binary);

    int meta_info_size = 16;
    std::vector<uint32_t> program_info(meta_info_size);
    /* read meta infos about the program */
    inst_file_stream.read(reinterpret_cast<char*>(program_info.data()), 4*meta_info_size);
    /* check header signature */
    if (program_info[0]!=2135247942)
    {
        std::cout << "WRONG BIN FILE!:" << std::endl;
        return EXIT_FAILURE;
    }

	cl_int hw_err;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue cmd_queue;
    cl::Kernel cu_krnl;

    std::string binaryFile(parser.value("xclbin_file"));
    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(hw_err, context = cl::Context(device, nullptr, nullptr, nullptr, &hw_err));
        OCL_CHECK(hw_err, cmd_queue = cl::CommandQueue(context, device, cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder, &hw_err));

        program = cl::Program(context, {device}, bins, nullptr, &hw_err);
        if (hw_err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            // std::cout << "Device[" << i << "]: program successful!\n";
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    std::string cu_id = std::to_string(1);
    std::string cu_krnls_name_full = std::string("cu_top")+ std::string(":{") + std::string("cu_top_") + cu_id + std::string("}");
    OCL_CHECK(hw_err, cu_krnl = cl::Kernel(program, cu_krnls_name_full.c_str(), &hw_err));

    int indice_mem_words = program_info[5] * DATA_PACK_NUM ;
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

	int hbmTotalChannels = program_info[12];

	int nnz_mem_pc_words = program_info[14];
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

	int lr_mem_pc_words = program_info[15];
	std::vector<float, aligned_allocator<float>> host_lr_buf(lr_mem_pc_words);

	std::vector<cl::Buffer> cu_rhs_mem(hbmTotalChannels);
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
                                              program_info[1]));
    OCL_CHECK(hw_err, hw_err = cu_krnl.setArg(krnlArgCount++,
                                              0));

    //Profiling
    double kernel_time_in_sec = 0;
    std::chrono::duration<double> kernel_time(0);
    auto kernel_start = std::chrono::high_resolution_clock::now();
    cmd_queue.enqueueTask(cu_krnl);
    cmd_queue.finish();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
    kernel_time_in_sec = kernel_time.count();
    /* run time in second */
    std::cout << "run time:  "<<
    std::setprecision(2) << std::scientific
    << kernel_time_in_sec<<"s"<<std::endl;

    /* Copy register file back */
    cmd_queue.enqueueMigrateMemObjects({cu_rhs_mem[0]},
                                       CL_MIGRATE_MEM_OBJECT_HOST);
    cmd_queue.finish();

    std::ofstream register_file("./temp/reg_content.txt");
    for (int i=0; i<64; i++){
        register_file<<host_lr_buf.data()[i]<<"\n";
    }
    register_file.close();

    /* Copy solution back, combining multiple HBM channels */
    int iscaC = DATA_PACK_NUM*hbmTotalChannels;
    int channelAddr = (program_info[9] >>1)*DATA_PACK_NUM; 
    int channelPacks = program_info[10];
    std::vector<float> combinedResult(channelPacks*iscaC);
    
	for(int hbmItem=0; hbmItem<hbmTotalChannels; hbmItem++) {
        /* Transfer HBM channel */
        cmd_queue.enqueueMigrateMemObjects({cu_rhs_mem[hbmItem]},
                                        CL_MIGRATE_MEM_OBJECT_HOST);
		cmd_queue.finish();
        /* Interleave Results */
        for(int pItem=0; pItem<channelPacks; pItem++){
            for(int dItem=0; dItem<DATA_PACK_NUM; dItem++){
                int srcAddr = channelAddr + pItem*DATA_PACK_NUM+dItem;
                int dstAddr = pItem*iscaC + hbmItem*DATA_PACK_NUM + dItem;
                combinedResult.data()[dstAddr] = host_lr_buf.data()[srcAddr];
            }
        }
    }
    save_results(combinedResult.data(), channelPacks*iscaC);

    return EXIT_SUCCESS;
}
