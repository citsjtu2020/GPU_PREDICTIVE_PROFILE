#
# Copyright 2021 NVIDIA Corporation. All rights reserved
#
ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

CUDA_INSTALL_PATH ?= /usr/local/cuda-11.8
PROFILER_HOST_UTILS_SRC ?= /usr/local/cuda-11.8/extras/CUPTI/samples/extensions/src/profilerhost_util
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"
INCLUDES := -I"$(CUDA_INSTALL_PATH)/include" -I/usr/local/cuda-11.8/extras/CUPTI/include -I/usr/local/cuda-11.8/extras/CUPTI/samples/extensions/include/profilerhost_util -I/usr/local/cuda-11.8/extras/CUPTI/samples/extensions/include/c_util

TARGET_ARCH ?= $(HOST_ARCH)
TARGET_OS ?= $(shell uname | tr A-Z a-z)

# Set required library paths.
# In the case of cross-compilation, set the libs to the correct ones under /usr/local/cuda/targets/<TARGET_ARCH>-<TARGET_OS>/lib

ifeq ($(OS), Windows_NT)
    LIB_PATH ?= ..\..\lib64
else
    ifneq ($(TARGET_ARCH), $(HOST_ARCH))
        INCLUDES += -I$(CUDA_INSTALL_PATH)/targets/$(HOST_ARCH)-$(shell uname | tr A-Z a-z)/include
        INCLUDES += -I$(CUDA_INSTALL_PATH)/targets/$(TARGET_ARCH)-$(TARGET_OS)/include
        LIB_PATH ?= $(CUDA_INSTALL_PATH)/targets/$(TARGET_ARCH)-$(TARGET_OS)/lib
        TARGET_CUDA_PATH = -L $(LIB_PATH)/stubs
    else
        EXTRAS_LIB_PATH := /usr/local/cuda-11.8/extras/CUPTI/lib64
        LIB_PATH ?= $(CUDA_INSTALL_PATH)/lib64
    endif
endif

ifeq ($(OS), Windows_NT)
    LIBS = -lcuda -L $(LIB_PATH) -lcupti -lnvperf_host -lnvperf_target -L ..\extensions\src\profilerhost_util -lprofilerHostUtil
    OBJ = obj
    LIBEXT = lib
    LIBPREFIX =
    BINEXT = .exe
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):$(LIB_PATH)
        LIBS = -Xlinker -framework -Xlinker cuda -L $(LIB_PATH) -lcupti -lnvperf_host -lnvperf_target -L /usr/local/cuda-11.8/extras/CUPTI/samples/extensions/src/profilerhost_util -lprofilerHostUtil
    else
        LIBS :=
        ifeq ($(HOST_ARCH), $(TARGET_ARCH))
            export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(LIB_PATH)
            LIBS = -L $(EXTRAS_LIB_PATH)
        endif
        LIBS += $(TARGET_CUDA_PATH) -lcuda -L $(LIB_PATH) -lcupti -lnvperf_host -lnvperf_target -L /usr/local/cuda-11.8/extras/CUPTI/samples/extensions/src/profilerhost_util -lprofilerHostUtil
    endif
    OBJ = o
    LIBEXT = a
    LIBPREFIX = lib
    BINEXT =
endif

# Gencode arguments
SMS ?= 70 72 75 80 86 87
# Generate SASS code for each SM architecture listed in $(SMS)
# Needed to support enhanced compatibility
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

.DEFAULT: all
.PHONY: all

all: profiler_host_util async_cupti_trace.so async_cupti_profile.so

profiler_host_util:
	cd $(PROFILER_HOST_UTILS_SRC) && $(MAKE)


async_cupti_trace.so: async_tracing_test.cpp
	$(NVCC) -o $@ $< $(INCLUDES) $(LIBS) -Ldl -Xcompiler -fPIC --shared --std c++14 `python3 -m pybind11 --includes`

async_cupti_profile.so: async_profiling_test.cpp
	$(NVCC) -o $@ $< $(INCLUDES) $(LIBS) -Ldl -Xcompiler -fPIC --shared --std c++14 `python3 -m pybind11 --includes`

.PHONY: clean
clean:
	rm -f async_cupti_trace.so async_cupti_profile.so
