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

HOST_SRCS += ./aws/includes/xcl2/xcl2.cpp
HOST_SRCS += ./aws/includes/cmdparser/cmdlineparser.cpp
HOST_SRCS += ./aws/includes/logger/logger.cpp 
HOST_SRCS += ./aws/rsqp.cpp
HOST_SRCS += ./aws/osqp_api.c

# Host compiler global settings
CXXFLAGS += -fmessage-length=0
LDFLAGS += -lrt -lstdc++ 

EXECUTABLE = ./rsqp

.PHONY: host
host: $(EXECUTABLE)

$(info $$CXXFLAGES is [${CXXFLAGS}])
$(info $$LDFLAGS is [${LDFLAGS}])
$(info $$HOST_SRCS is [${HOST_SRCS}])
$(EXECUTABLE): $(HOST_SRCS) 
		$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)
