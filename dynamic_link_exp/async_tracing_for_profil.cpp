#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#include <stdlib.h>

#include <map>
using ::std::map;


#include "cuda_runtime_api.h"
#include "cupti_callbacks.h"
#include "cupti_profiler_target.h"
// #include "cupti_driver_cbid.h"
#include "cupti_target.h"
#include "cupti_activity.h"
#include "nvperf_host.h"

#include <iostream>
using ::std::cerr;
using ::std::cout;
using ::std::endl;

#include <mutex>
using ::std::mutex;

#include <string>
using ::std::string;

#include <vector>
using ::std::vector;

#include <unordered_map>
using ::std::unordered_map;

#include <sstream>
using ::std::ostringstream;

#include <sys/stat.h> // for mkdir on Linux/Unix
//#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>
//#include <Metric.h>
//#include <Eval.h>
#include <FileOp.h>
#include <fstream>

# include <pybind11/pybind11.h>

namespace py = pybind11;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#define CUPTI_CALL(call)                                                    \
  do {                                                                      \
    CUptiResult _status = call;                                             \
    if (_status != CUPTI_SUCCESS) {                                         \
      const char *errstr;                                                   \
      cuptiGetResultString(_status, &errstr);                               \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
              __FILE__, __LINE__, #call, errstr);                           \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define MEMORY_ALLOCATION_CALL(var)                                            \
do {                                                                            \
    if (var == NULL) {                                                          \
        fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n",            \
                __FILE__, __LINE__);                                            \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)



#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

static uint64_t startTimestamp1;

bool directoryExists(string &path){
    struct stat info;
    if (stat(path.c_str(),&info)!=0){
        return false;
    }else{
        return true;
    }
}

bool createDirectory(string &path){
    mode_t mode = 0755;
    bool exist_check = directoryExists(path);
    if (exist_check){
        return true;
    }

    int result = mkdir(path.c_str(),mode);

    return (result == 0);
}

typedef struct{
    uint64_t start;
    uint32_t correlationId;
    string kernelName;
    uint32_t kernelLaunchId;
}kernel_trace_info;

typedef struct{
    uint64_t start;
    uint32_t cbid;
    uint32_t correlationId;
}api_trace_info;

typedef struct {
    volatile uint32_t initialized;
    // CUpti_SubscriberHandle  subscriber;
    volatile uint32_t detachCupti;
    int frequency;
    int tracingEnabled;
    int terminateThread;
    uint64_t kernelsTraced;
    uint64_t objectTraced;
    uint64_t runtimeAPITraced;
    uint64_t driverAPITraced;
    uint64_t deviceTraced;
    uint64_t contextTraced;
    uint64_t memcpyTraced;
    uint64_t memsetTraced;
    uint64_t overheadTraced;
    vector<string> tracingOutBuffer;
    vector<kernel_trace_info> tracingKernelLaunch;
    unordered_map<uint32_t,api_trace_info> runtimeCorr;
    unordered_map<uint32_t,api_trace_info> driverCorr;

    mutex trace_data_mutex;
    // pthread_t dynamicThread;
    // pthread_mutex_t mutexFinalize;
    // pthread_cond_t mutexCondition;
} injGlobalControl;

injGlobalControl globalControl;

static void
globalControlInit(void) {
    globalControl.initialized = 0;
    // globalControl.subscriber = 0;
    globalControl.detachCupti = 0;
    globalControl.frequency = 2; // in seconds
    globalControl.tracingEnabled = 0;
    globalControl.terminateThread = 0;
    globalControl.kernelsTraced = 0;
    globalControl.objectTraced = 0;
    globalControl.runtimeAPITraced = 0;
    globalControl.driverAPITraced = 0;
    globalControl.deviceTraced = 0;
    globalControl.contextTraced = 0;
    globalControl.memcpyTraced = 0;
    globalControl.memsetTraced = 0;
    globalControl.overheadTraced = 0;

    // globalControl.mutexFinalize = PTHREAD_MUTEX_INITIALIZER;
    // globalControl.mutexCondition = PTHREAD_COND_INITIALIZER;
}



static const char *
getMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
    return "HtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
    return "DtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
    return "HtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
    return "AtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
    return "AtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
    return "AtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
    return "DtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
    return "DtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
    return "HtoH";
  default:
    break;
  }

  return "<unknown>";
}

const char *
getActivityOverheadKindString(CUpti_ActivityOverheadKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
    return "COMPILER";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
    return "BUFFER_FLUSH";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
    return "INSTRUMENTATION";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
    return "RESOURCE";
  default:
    break;
  }

  return "<unknown>";
}

const char *
getActivityObjectKindString(CUpti_ActivityObjectKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return "PROCESS";
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return "THREAD";
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return "DEVICE";
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return "CONTEXT";
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return "STREAM";
  default:
    break;
  }

  return "<unknown>";
}

uint32_t
getActivityObjectKindId(CUpti_ActivityObjectKind kind, CUpti_ActivityObjectKindId *id)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return id->pt.processId;
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return id->pt.threadId;
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return id->dcs.deviceId;
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return id->dcs.contextId;
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return id->dcs.streamId;
  default:
    break;
  }

  return 0xffffffff;
}

bool store_tracing_results(string filepath){
    bool mk_path_res = createDirectory(filepath);

    if(mk_path_res){
        //        int session_iter = ctx_data.iterations;
        //        uint32_t context_id = 1;
        //        if (ProfilerControl.ctx_id.count(ctx_data.ctx)){
        //            context_id = ProfilerControl.ctx_id[ctx_data.ctx];
        //        }else{
        //            context_id = 1;
        //        }

//        string out_results = "Kernel,context,globalId,start,end,duration\n";
//        int session_iter = ctx_data.iterations;
        string out_results = "";

        for(auto it=globalControl.tracingOutBuffer.begin();it!=globalControl.tracingOutBuffer.end();it++){
            string tmp_Res = *it;
            out_results = out_results + tmp_Res;
        }

        string kernel_info_results = "kernel,launchId,start,correlationId\n";
        //vector<kernel_trace_info> tracingKernelLaunch;
        //unordered_map<uint32_t,api_trace_info> runtimeCorr;
        //unordered_map<uint32_t,api_trace_info> driverCorr;
        map<uint64_t,kernel_trace_info> sorted_map_kti;
        for(auto it=globalControl.tracingKernelLaunch.begin();it!=globalControl.tracingKernelLaunch.end();it++){
            //uint64_t start;
            //uint32_t correlationId;
            //string kernelName;
            //uint32_t kernelLaunchId;

            kernel_trace_info kti_instance = *it;
            uint32_t tmp_corr_id = kti_instance.correlationId;
            //uint64_t start;
            //unit32_t cbid;
            //uint32_t correlationId;
            //}api_trace_info;
            if (globalControl.driverCorr.count(tmp_corr_id) > 0){
                api_trace_info ati_instance = globalControl.driverCorr[tmp_corr_id];
                kti_instance.start = ati_instance.start;
                sorted_map_kti[kti_instance.start] = kti_instance;
            }else{
                if (globalControl.runtimeCorr.count(tmp_corr_id)>0){
                    api_trace_info ati_instance = globalControl.runtimeCorr[tmp_corr_id];
                    kti_instance.start = ati_instance.start;
                    sorted_map_kti[kti_instance.start] = kti_instance;
                }
            }
        }

        uint32_t total_launch_id = 0;
        vector<string> sorted_output_kti;
        //"name,launchId,start,correlationId"
        for (auto& pair : sorted_map_kti) {
            kernel_trace_info kti_instance = pair.second;
            kti_instance.kernelLaunchId = total_launch_id;
            total_launch_id = total_launch_id + 1;
            string tmp_string = kti_instance.kernelName+","+std::to_string(kti_instance.kernelLaunchId)+","+std::to_string(kti_instance.start)+","+std::to_string(kti_instance.correlationId);
            kernel_info_results = kernel_info_results + tmp_string + "\n";
        }

        // +std::to_string(startTimestamp1)
        string base_file_name = filepath + "/" + "trace_raw";
        string out_file_name = base_file_name+".log";

        string kernel_file_name = filepath + "/" + "kernel_launch_info.csv";

        std::ofstream outFile(out_file_name);

        if (outFile.is_open()){
            outFile << out_results;
            outFile.close();
            std::cout <<"Trace Data written successfully!" << std::endl;
        }else{
            std::cerr<<"Unable to open file for writing."<<std::endl;
            return false;
        }

        std::ofstream outLaunch(kernel_file_name);

        if (outLaunch.is_open()){
            outLaunch << kernel_info_results;
            outLaunch.close();
            std::cout <<"Kernel Launch Info written successfully!" << std::endl;
        }else{
            std::cerr<<"Unable to open file for writing."<<std::endl;
            return false;
        }

        return true;
    }else{
        std::cerr<<"Path Not Exists: Unable to create file for writing."<<std::endl;
        return false;
    }


}

static void
printSummary(void) {

    printf("\n-------------------------------------------------------------------\n");
    printf("\tKernels traced : %llu", (unsigned long long)globalControl.kernelsTraced);
    printf("\tObject traced : %llu", (unsigned long long)globalControl.objectTraced);
    printf("\tRuntime API traced : %llu", (unsigned long long)globalControl.runtimeAPITraced);
    printf("\tDriver API traced : %llu", (unsigned long long)globalControl.driverAPITraced);
    printf("\tContext traced : %llu", (unsigned long long)globalControl.contextTraced);
    printf("\tMEMCPY traced : %llu", (unsigned long long)globalControl.memcpyTraced);
    printf("\tMEMSET traced : %llu", (unsigned long long)globalControl.memsetTraced);
    printf("\n-------------------------------------------------------------------\n");

    std::ostringstream out_summ_info;
    out_summ_info << "\n-------------------------------------------------------------------\n"
    <<"\tKernels traced : "<<((unsigned long long)globalControl.kernelsTraced)
    <<"\tObject traced : "<<((unsigned long long)globalControl.objectTraced)
    <<"\tRuntime API traced : "<<((unsigned long long)globalControl.runtimeAPITraced)
    <<"\tDriver API traced : "<<((unsigned long long)globalControl.driverAPITraced)
    <<"\tContext traced : "<<((unsigned long long)globalControl.contextTraced)
    <<"\tMEMCPY traced : "<<((unsigned long long)globalControl.memcpyTraced)
    <<"\tMEMSET traced : "<<((unsigned long long)globalControl.memsetTraced)
    <<"\n-------------------------------------------------------------------\n";

      std::string result = out_summ_info.str();
      globalControl.trace_data_mutex.lock();
      globalControl.tracingOutBuffer.push_back(result);
      globalControl.trace_data_mutex.unlock();


}

static void
atExitHandler(void) {
    globalControl.terminateThread = 1;



    // Force flush
    if(globalControl.tracingEnabled) {
        CUPTI_CALL(cuptiActivityFlushAll(1));
    }

    // PTHREAD_CALL(pthread_join(globalControl.dynamicThread, NULL));

}

void registerAtExitHandler(void) {
    // Register atExitHandler
    atexit(&atExitHandler);
}


static const char *
getComputeApiKindString(CUpti_ActivityComputeApiKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_COMPUTE_API_CUDA:
    return "CUDA";
  case CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS:
    return "CUDA_MPS";
  default:
    break;
  }

  return "<unknown>";
}

static void
printActivity(CUpti_Activity *record)
{
  switch (record->kind)
  {
  case CUPTI_ACTIVITY_KIND_DEVICE:
    {
      CUpti_ActivityDevice4 *device = (CUpti_ActivityDevice4 *) record;
      printf("DEVICE %s (%u), capability %u.%u, global memory (bandwidth %u GB/s, size %u MB), "
             "multiprocessors %u, clock %u MHz\n",
             device->name, device->id,
             device->computeCapabilityMajor, device->computeCapabilityMinor,
             (unsigned int) (device->globalMemoryBandwidth / 1024 / 1024),
             (unsigned int) (device->globalMemorySize / 1024 / 1024),
             device->numMultiprocessors, (unsigned int) (device->coreClockRate / 1000));
      std::ostringstream out_device_info;
      out_device_info << "DEVICE " << device->name<< " (" <<device->id<<")"<<", capability "
      <<device->computeCapabilityMajor<<"."<<device->computeCapabilityMinor<<", global memory (bandwidth "
      <<((unsigned int) (device->globalMemoryBandwidth / 1024 / 1024))<<" GB/s, size "
      <<((unsigned int) (device->globalMemorySize / 1024 / 1024))<<" MB), "<<"multiprocessors "<<device->numMultiprocessors
      <<", clock "<<((unsigned int) (device->coreClockRate / 1000))<<" MHz\n";

      std::string result = out_device_info.str();
      globalControl.trace_data_mutex.lock();
      globalControl.tracingOutBuffer.push_back(result);
      globalControl.trace_data_mutex.unlock();
      globalControl.deviceTraced++;
      break;
    }
  case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
    {
      CUpti_ActivityDeviceAttribute *attribute = (CUpti_ActivityDeviceAttribute *)record;
      printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n",
             attribute->attribute.cupti, attribute->deviceId, (unsigned long long)attribute->value.vUint64);
      std::ostringstream out_device_info;
      out_device_info << "DEVICE_ATTRIBUTE " << attribute->attribute.cupti<< ", device "
      <<attribute->deviceId<<", value="<<((unsigned long long)attribute->value.vUint64)
      <<"\n";

      std::string result = out_device_info.str();
      globalControl.trace_data_mutex.lock();
      globalControl.tracingOutBuffer.push_back(result);
      globalControl.trace_data_mutex.unlock();
      break;
    }
  case CUPTI_ACTIVITY_KIND_CONTEXT:
    {
      CUpti_ActivityContext *context = (CUpti_ActivityContext *) record;
      printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n",
             context->contextId, context->deviceId,
             getComputeApiKindString((CUpti_ActivityComputeApiKind) context->computeApiKind),
             (int) context->nullStreamId);
      std::ostringstream out_context_info;
      out_context_info << "CONTEXT " << context->contextId<< ", device "
      <<context->deviceId<<", compute API "<<getComputeApiKindString((CUpti_ActivityComputeApiKind) context->computeApiKind)
      <<"NULL stream "<<((int) context->nullStreamId)
      <<"\n";

      std::string result = out_context_info.str();
      globalControl.trace_data_mutex.lock();
      globalControl.tracingOutBuffer.push_back(result);
      globalControl.trace_data_mutex.unlock();
      globalControl.contextTraced++;
      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMCPY:
    {
      CUpti_ActivityMemcpy5 *memcpy = (CUpti_ActivityMemcpy5 *) record;
      printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, size %llu, correlation %u\n",
              getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind),
              (unsigned long long) (memcpy->start - startTimestamp1),
              (unsigned long long) (memcpy->end - startTimestamp1),
              memcpy->deviceId, memcpy->contextId, memcpy->streamId,
              (unsigned long long)memcpy->bytes, memcpy->correlationId);

      std::ostringstream out_memcpy_info;
      out_memcpy_info << "MEMCPY " << getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind)
      << "[ "
      <<((unsigned long long) (memcpy->start - startTimestamp1))
      <<" - "
      <<((unsigned long long) (memcpy->end - startTimestamp1))
      <<" ] device "<<memcpy->deviceId<<", context "<<memcpy->contextId
      <<", stream "<<memcpy->streamId<<", size "<<((unsigned long long)memcpy->bytes)
      <<", correlation "<<memcpy->correlationId<<"\n";

      std::string result = out_memcpy_info.str();
      globalControl.trace_data_mutex.lock();
      globalControl.tracingOutBuffer.push_back(result);
      globalControl.trace_data_mutex.unlock();
      globalControl.memcpyTraced++;
      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMSET:
    {
      CUpti_ActivityMemset4 *memset = (CUpti_ActivityMemset4 *) record;
      printf("MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, size %llu correlation %u\n",
             memset->value,
            ((unsigned long long) (memset->start - startTimestamp1)),
             ((unsigned long long) (memset->end - startTimestamp1)),
             memset->deviceId, memset->contextId, memset->streamId,(unsigned long long)memset->bytes,
             memset->correlationId);
      std::ostringstream out_memset_info;
      out_memset_info << "MEMSET value=" << memset->value
      << "[ "
      <<((unsigned long long) (memset->start - startTimestamp1))
      <<" - "
      <<((unsigned long long) (memset->end - startTimestamp1))
      <<" ] device "<<memset->deviceId<<", context "<<memset->contextId
      <<", stream "<<memset->streamId
      <<", size "<<((unsigned long long)memset->bytes)
      <<", correlation "<<memset->correlationId<<"\n";

      std::string result = out_memset_info.str();
      globalControl.trace_data_mutex.lock();
      globalControl.tracingOutBuffer.push_back(result);
      globalControl.trace_data_mutex.unlock();
      globalControl.memsetTraced++;
      break;
    }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    //  - startTimestamp1
    // - startTimestamp1
    {
      const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
      CUpti_ActivityKernel8 *kernel = (CUpti_ActivityKernel8 *) record;
      printf("%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
             kindString,
             kernel->name,
             (unsigned long long) (kernel->start),
             (unsigned long long) (kernel->completed),
             kernel->deviceId, kernel->contextId, kernel->streamId,
             kernel->correlationId);

      printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
             kernel->gridX, kernel->gridY, kernel->gridZ,
             kernel->blockX, kernel->blockY, kernel->blockZ,
             kernel->staticSharedMemory, kernel->dynamicSharedMemory);
      //       - startTimestamp1 - startTimestamp1
      std::ostringstream out_kernel_info;
      out_kernel_info << kindString
      << "\""
      <<kernel->name
      <<"\" [ "
      <<((unsigned long long) (kernel->start))
      <<" - "
      <<((unsigned long long) (kernel->completed))
      <<" ] device "<<kernel->deviceId<<", context "<<kernel->contextId
      <<", stream "<<kernel->streamId
      <<", correlation "<<kernel->correlationId
      <<", grid ["<<kernel->gridX<<","<<kernel->gridY<<","
      <<kernel->gridZ<<"], block ["
      <<kernel->blockX<<","<<kernel->blockY<<","<<kernel->blockZ<<"]"
      <<", shared memory (static "<<kernel->staticSharedMemory
      <<", dynamic "<<kernel->dynamicSharedMemory<<")\n";

      //uint64_t start;
      //uint32_t correlationId;
      //string kernelName;
      //uint32_t kernelLaunchId;
      kernel_trace_info kti_instance = {};
      kti_instance.start = 0;
      kti_instance.correlationId = kernel -> correlationId;
      kti_instance.kernelName = kernel->name;
      kti_instance.kernelLaunchId = 0;

      std::string result = out_kernel_info.str();
      globalControl.trace_data_mutex.lock();
      globalControl.tracingOutBuffer.push_back(result);
      globalControl.tracingKernelLaunch.push_back(kti_instance);
      globalControl.trace_data_mutex.unlock();


      globalControl.kernelsTraced++;
      break;
    }
  case CUPTI_ACTIVITY_KIND_DRIVER:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      printf("DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
             api->cbid,
             (unsigned long long) (api->start - startTimestamp1),
             (unsigned long long) (api->end - startTimestamp1),
             api->processId, api->threadId, api->correlationId);
      std::ostringstream out_driver_info;
      out_driver_info<< "DRIVER cbid="
      <<api->cbid
      <<" [ "
      <<((unsigned long long) (api->start - startTimestamp1))
      <<" - "
      <<((unsigned long long) (api->end - startTimestamp1))
      <<" ] process "<<api->processId<<", thread "<<api->threadId
      <<", correlation "<<api->correlationId<<"\n";

      api_trace_info ati_instance = {};
      ati_instance.start = ((unsigned long long) (api->start - startTimestamp1));
      ati_instance.cbid = api->cbid;
      ati_instance.correlationId = api->correlationId;

      std::string result = out_driver_info.str();
      globalControl.trace_data_mutex.lock();
      globalControl.tracingOutBuffer.push_back(result);

      globalControl.driverCorr[api->correlationId] = ati_instance;
      globalControl.trace_data_mutex.unlock();
      globalControl.driverAPITraced++;
      break;
    }
  case CUPTI_ACTIVITY_KIND_RUNTIME:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
             api->cbid,
             (unsigned long long) (api->start - startTimestamp1),
             (unsigned long long) (api->end - startTimestamp1),
             api->processId, api->threadId, api->correlationId);

      std::ostringstream out_runtime_info;
      out_runtime_info<< "RUNTIME cbid="
      <<api->cbid
      <<" [ "
      <<((unsigned long long) (api->start - startTimestamp1))
      <<" - "
      <<((unsigned long long) (api->end - startTimestamp1))
      <<" ] process "<<api->processId<<", thread "<<api->threadId
      <<", correlation "<<api->correlationId<<"\n";

      std::string result = out_runtime_info.str();

      api_trace_info ati_instance = {};
      ati_instance.start = (unsigned long long)(api->start - startTimestamp1);
      ati_instance.cbid = api->cbid;
      ati_instance.correlationId = api->correlationId;


      globalControl.trace_data_mutex.lock();
      globalControl.tracingOutBuffer.push_back(result);
      globalControl.runtimeCorr[api->correlationId] = ati_instance;
      globalControl.trace_data_mutex.unlock();

      globalControl.runtimeAPITraced++;
      break;
    }
  case CUPTI_ACTIVITY_KIND_NAME:
    {
      CUpti_ActivityName *name = (CUpti_ActivityName *) record;
      globalControl.objectTraced++;
      std::ostringstream out_name_info;
      std::string result;
      switch (name->objectKind)
      {
      case CUPTI_ACTIVITY_OBJECT_CONTEXT:
        printf("NAME  %s %u %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
               name->name);

        out_name_info<< "NAME "
        <<getActivityObjectKindString(name->objectKind)<<" "
        <<getActivityObjectKindId(name->objectKind, &name->objectId)<<" "
        <<getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE)<<" "
        <<"id "<<getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId)<<", "
        <<"name "<< name->name
        <<"\n";

        result = out_name_info.str();
        globalControl.trace_data_mutex.lock();
        globalControl.tracingOutBuffer.push_back(result);
        globalControl.trace_data_mutex.unlock();

        break;

      case CUPTI_ACTIVITY_OBJECT_STREAM:
        printf("NAME %s %u %s %u %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_CONTEXT),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_CONTEXT, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
               name->name);


        out_name_info<< "NAME "
        <<getActivityObjectKindString(name->objectKind)<<" "
        <<getActivityObjectKindId(name->objectKind, &name->objectId)<<" "
        <<getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_CONTEXT)<<" "
        <<getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_CONTEXT, &name->objectId)<<" "
        <<getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE)<<" "
        <<"id "<<getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId)<<", "
        <<"name "<< name->name
        <<"\n";

        result = out_name_info.str();
        globalControl.trace_data_mutex.lock();
        globalControl.tracingOutBuffer.push_back(result);
        globalControl.trace_data_mutex.unlock();

        break;
      default:
        printf("NAME %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               name->name);

        out_name_info<< "NAME "
        <<getActivityObjectKindString(name->objectKind)<<" "
        <<"id "<<getActivityObjectKindId(name->objectKind, &name->objectId)<<", "
        <<"name "<< name->name
        <<"\n";

        result = out_name_info.str();
        globalControl.trace_data_mutex.lock();
        globalControl.tracingOutBuffer.push_back(result);
        globalControl.trace_data_mutex.unlock();
        break;
      }
      break;
    }
  case CUPTI_ACTIVITY_KIND_MARKER:
    {
      CUpti_ActivityMarker2 *marker = (CUpti_ActivityMarker2 *) record;
      printf("MARKER id %u [ %llu ], name %s, domain %s\n",
             marker->id, (unsigned long long) marker->timestamp, marker->name, marker->domain);
      std::ostringstream out_marker_info;
      out_marker_info<< "MARKER id "
      <<marker->id<<" "
      <<" [ "<<((unsigned long long) marker->timestamp)<<" ], name "
      <<marker->name
      <<", domain "
      <<marker->domain
      <<"\n";

      std::string result = out_marker_info.str();
      globalControl.trace_data_mutex.lock();
      globalControl.tracingOutBuffer.push_back(result);
      globalControl.trace_data_mutex.unlock();
      break;
    }
  case CUPTI_ACTIVITY_KIND_MARKER_DATA:
    {
      CUpti_ActivityMarkerData *marker = (CUpti_ActivityMarkerData *) record;
      printf("MARKER_DATA id %u, color 0x%x, category %u, payload %llu/%f\n",
             marker->id, marker->color, marker->category,
             (unsigned long long) marker->payload.metricValueUint64,
             marker->payload.metricValueDouble);
      std::ostringstream out_marker_info;
      out_marker_info<< "MARKER_DATA id "
      <<marker->id<<" "
      <<", color "<<marker->color<<", category "
      <<marker->category
      <<", payload "
      <<((unsigned long long) marker->payload.metricValueUint64)
      <<"/"<<marker->payload.metricValueDouble
      <<"\n";

      std::string result = out_marker_info.str();
      globalControl.trace_data_mutex.lock();
      globalControl.tracingOutBuffer.push_back(result);
      globalControl.trace_data_mutex.unlock();
      break;
    }
  case CUPTI_ACTIVITY_KIND_OVERHEAD:
    {
      CUpti_ActivityOverhead *overhead = (CUpti_ActivityOverhead *) record;
      globalControl.overheadTraced;
      printf("OVERHEAD %s [ %llu, %llu ] %s id %u\n",
             getActivityOverheadKindString(overhead->overheadKind),
             (unsigned long long) overhead->start - startTimestamp1,
             (unsigned long long) overhead->end - startTimestamp1,
             getActivityObjectKindString(overhead->objectKind),
             getActivityObjectKindId(overhead->objectKind, &overhead->objectId));

      std::ostringstream out_overhead_info;
      out_overhead_info<< "OVERHEAD "
      <<getActivityOverheadKindString(overhead->overheadKind)<<" "
      <<"[ "<<(unsigned long long)((unsigned long long) overhead->start - startTimestamp1)<<", "
      <<(unsigned long long)((unsigned long long) overhead->end - startTimestamp1)
      <<" ] "
      <<getActivityObjectKindString(overhead->objectKind)
      <<" id "<<getActivityObjectKindId(overhead->objectKind, &overhead->objectId)
      <<"\n";

      std::string result = out_overhead_info.str();
      globalControl.trace_data_mutex.lock();
      globalControl.tracingOutBuffer.push_back(result);
      globalControl.trace_data_mutex.unlock();
      break;
    }
  default:
    printf("  <unknown>\n");
    break;
  }
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(EXIT_FAILURE);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        printActivity(record);
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      printf("Dropped %u activity records\n", (unsigned int) dropped);
    }
  }

  free(buffer);
}

static CUptiResult
cuptiInitialize(void) {
    CUptiResult status = CUPTI_SUCCESS;

    // CUPTI_CALL(cuptiSubscribe(&globalControl.subscriber, (CUpti_CallbackFunc)callbackHandler, NULL));

    // Subscribe Driver and Runtime callbacks to call cuptiFinalize in the entry/exit callback of these APIs
    // CUPTI_CALL(cuptiEnableDomain(1, globalControl.subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    // CUPTI_CALL(cuptiEnableDomain(1, globalControl.subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

    // Device activity record is created when CUDA initializes, so we
    // want to enable it before cuInit() or any CUDA runtime call.
//    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));

    // Enable CUPTI activities (Enable all other activity record kinds.)
//    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
//    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
//    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
//    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
//    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));

    // Register buffer callbacks
    // CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    size_t attrValue = 0, attrValueSize = sizeof(size_t);

    // Register callbacks for buffer requests and for buffers completed by CUPTI.
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    // Optionally get and set activity attributes.
    // Attributes can be set by the CUPTI client to change behavior of the activity API.
    // Some attributes require to be set before any CUDA context is created to be effective,
    // e.g. to be applied to all device buffer allocations (see documentation).

    CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));

    printf("%s = %llu B\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned)attrValue);
    attrValue *= 2;

    CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));

    CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize,&attrValue));
    printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned)attrValue);
    attrValue *= 2;
    CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize,&attrValue));
    CUPTI_CALL(cuptiGetTimestamp(&startTimestamp1));
    printf("CUPTI_START_TIMESTAMP = %llu\n",(unsigned long long)startTimestamp1);
    std::ostringstream out_start_info;

    out_start_info<< "CUPTI_START_TIMESTAMP = "
    <<(unsigned long long)(startTimestamp1)
    <<"\n";

    std::string result = out_start_info.str();
    globalControl.trace_data_mutex.lock();
    globalControl.tracingOutBuffer.push_back(result);
    globalControl.trace_data_mutex.unlock();



    return status;
}

int InitializeTrace(void) {

    if (globalControl.initialized) {
        return 1;
    }
    // Init globalControl
    globalControlInit();

    // //Initialize Mutex
    // PTHREAD_CALL(pthread_mutex_init(&globalControl.mutexFinalize, 0));

    registerAtExitHandler();

    // Initialize CUPTI
    CUPTI_CALL(cuptiInitialize());
    globalControl.tracingEnabled = 1;

    // // Launch the thread
    // PTHREAD_CALL(pthread_create(&globalControl.dynamicThread, NULL, dynamicAttachDetach, NULL));

    globalControl.initialized = 1;

    return 1;
}

void FiniTrace(string filepath)
{
   // Force flush any remaining activity buffers before termination of the application
   CUPTI_CALL(cuptiActivityFlushAll(1));
//   CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONTEXT));
   CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
   CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
   CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
   CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
//   CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_NAME));
//   CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MARKER));
   CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
//   CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
//   CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_OVERHEAD));
   // cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
   globalControl.detachCupti = 1;
   globalControl.terminateThread=1;
   globalControl.tracingEnabled = 0;
   printSummary();
   if (filepath.empty()){
       std::cout<<"do not store results"<<std::endl;
   }else{
       store_tracing_results(filepath);
   }

   CUPTI_CALL(cuptiFinalize());
}



PYBIND11_MODULE(async_cupti_trace_for_pro, m){
    m.doc() = "pybind11 for async cupti trace profiling";
    m.def("InitializeTrace", &InitializeTrace, "A function which starts a tracing procedure");
    m.def("FiniTrace",&FiniTrace,"A function which destories a tracing procedure");
    // m.def("inadd", &inadd, "cin and cout");
}




