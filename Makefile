CXX := g++

xrt_path = $(XILINX_XRT)
OPENCL_INCLUDE:= $(xrt_path)/include

VIVADO_INCLUDE:= $(XILINX_VIVADO)/include
opencl_CXXFLAGS=-I$(OPENCL_INCLUDE) -I$(VIVADO_INCLUDE)
OPENCL_LIB:= $(xrt_path)/lib
opencl_LDFLAGS=-L$(OPENCL_LIB) -lOpenCL -pthread

CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++1y
LDFLAGS += $(opencl_LDFLAGS)

#Include Required Host Source Files
CXXFLAGS += -I./aws/includes/xcl2
CXXFLAGS += -I./aws/includes/cmdparser
CXXFLAGS += -I./aws/includes/logger
CXXFLAGS += -I./aws/includes

FPGA_SRCS += ./aws/includes/xcl2/xcl2.cpp
FPGA_SRCS += ./aws/includes/cmdparser/cmdlineparser.cpp
FPGA_SRCS += ./aws/includes/logger/logger.cpp 

HOST_SRCS += ./aws/rsqp.cpp
EXECUTABLE = ./rsqp

DEMO_SRCS += ./aws/osqp_api.c
DEMO_SRCS += ./aws/osqp_demo.c
DEMO_EXE = ./osqp_demo
DEMO_INPUT = ./temp/demo_input.dat

# Host compiler global settings
CXXFLAGS += -fmessage-length=0
LDFLAGS += -lrt -lstdc++ 

# $(info $$CXXFLAGES is [${CXXFLAGS}])
.PHONY: host

# Original Vitis host program
host: $(EXECUTABLE)
$(EXECUTABLE): $(FPGA_SRCS) $(HOST_SRCS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

# C API for the FPGA backend
demo: $(DEMO_EXE)
$(DEMO_EXE): $(FPGA_SRCS) $(DEMO_SRCS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

input:
	python ./aws/runtime_data_utils.py

run: demo
	$(DEMO_EXE) $(DEMO_INPUT)
