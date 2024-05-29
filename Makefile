#
# Copyright 2019-2021 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# makefile-generator v1.0.3
#

#Setting CXX
CXX := g++
HOST_ARCH := x86

include ./aws/includes/opencl/opencl.mk
CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++1y
LDFLAGS += $(opencl_LDFLAGS)

############################## Setting up Host Variables ##############################
#Include Required Host Source Files
CXXFLAGS += -I./aws/includes/xcl2
CXXFLAGS += -I./aws/includes/cmdparser
CXXFLAGS += -I./aws/includes/logger
HOST_SRCS += ./aws/includes/xcl2/xcl2.cpp \
		./aws/includes/cmdparser/cmdlineparser.cpp \
		./aws/includes/logger/logger.cpp \
		./aws/rsqp.cpp
# Host compiler global settings
CXXFLAGS += -fmessage-length=0
LDFLAGS += -lrt -lstdc++ 

############################## Setting up Kernel Variables ##############################

EXECUTABLE = ./rsqp

.PHONY: host
host: $(EXECUTABLE)

$(info $$CXXFLAGES is [${CXXFLAGS}])
$(info $$LDFLAGS is [${LDFLAGS}])
$(info $$HOST_SRCS is [${HOST_SRCS}])
$(EXECUTABLE): $(HOST_SRCS) 
		$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)
