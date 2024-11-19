#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <cupti_callbacks.h>
#include <cupti_driver_cbid.h>
#include <nvperf_host.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_runtime_api.h"
#include "cupti_target.h"
#include "cupti_activity.h"

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h> // for mkdir on Linux/Unix
//#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>
//#include <Metric.h>
//#include <Eval.h>
#include <FileOp.h>
#include <fstream>


#include <stdio.h>
//#include <stdlib.h>
#include <Eval.h>
using ::NV::Metric::Eval::PrintMetricValues;

#include <Metric.h>
using ::NV::Metric::Config::GetConfigImage;
using ::NV::Metric::Config::GetCounterDataPrefixImage;

#include <Utils.h>
using ::NV::Metric::Utils::GetNVPWResultString;

#include <iostream>
using ::std::cerr;
using ::std::cout;
using ::std::endl;

#include <mutex>
using ::std::mutex;

#include <string>
using ::std::string;

#include <unordered_map>
using ::std::unordered_map;

#include <unordered_set>
using ::std::unordered_set;

#include <vector>
using ::std::vector;

# include <pybind11/pybind11.h>

namespace py = pybind11;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif



#define NVPW_API_CALL(apiFuncCall)                                             \
do {                                                                           \
    NVPA_Status _status = apiFuncCall;                                         \
    if (_status != NVPA_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

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

// Helpful error handlers for standard CUPTI and CUDA runtime calls
#define CUPTI_API_CALL(apiFuncCall)                                            \
do {                                                                           \
    CUptiResult _status = apiFuncCall;                                         \
    if (_status != CUPTI_SUCCESS) {                                            \
        const char *errstr;                                                    \
        cuptiGetResultString(_status, &errstr);                                \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, errstr);                     \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

struct ProfilingData_t
{

    bool bProfiling = false;
    std::string chipName;
    std::vector<std::string> metricNames;

    CUpti_ProfilerRange profilerRange = CUPTI_AutoRange;
    CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_UserReplay;
    bool allPassesSubmitted = true;
    std::vector<uint8_t> counterDataImagePrefix;
    std::vector<uint8_t> configImage;
    std::vector<uint8_t> counterDataImage;
    std::vector<uint8_t> counterDataScratchBuffer;

};

// Profiler API configuration data, per-context
struct ctxProfilerData
{
    CUcontext       ctx;
    int             dev_id;
    std::string     chipName;
    vector<uint8_t> counterAvailabilityImage;
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
    vector<uint8_t> counterDataImage;
    vector<uint8_t> counterDataPrefixImage;
    vector<uint8_t> counterDataScratchBufferImage;
    vector<uint8_t> configImage;
    //    unordered_map<int,CallbackKernelTimestamp_t> kerneltime_data;
    int             maxNumRanges;
    int             curRanges;
    int             maxRangeNameLength;
    bool            session_started = false;
    bool            pass_started = false;
    bool            range_started = false;
    bool            profile_enabled = false;
    int             iterations; // Count of Sessions
    bool allPassesSubmitted = true;
};

//struct CallbackKernelTimestamp_t{
//    string kernelName;
//    uint32_t contextId;
//    uint64_t startTimestamp;
//    uint64_t endTimestamp;
//    uint64_t kernel_duration;
//};

//SimpleCupti.counterdata
class GlobalProfilerControl_t{
    public:
        int numRanges = 2;
        int nowContextId = 1;
        int kernelPerRange = 1;
        int numPassPerSess = 1;
        int initialized = 0;
        string CounterDataFilePath = "SimpleCupti";
        CUpti_SubscriberHandle  subscriber;
        // Track per-context profiler API data in a shared map
        mutex ctx_data_mutex;
        unordered_map<CUcontext, ctxProfilerData> ctx_data;

        //    uint32_t contextUid;
        unordered_map<CUcontext,uint32_t> ctx_id;
        // List of metrics to collect
        vector<string> metricNames;
        int maxRangeNameLength=64;
        int auto_range=1;
        int kernel_replay=1;

        //    string CounterDataSBFileName = "SimpleCupti.counterdataSB";
        GlobalProfilerControl_t() {}
};

//GlobalProfilerControl_t

GlobalProfilerControl_t ProfilerControl;

static uint64_t startTimestamp;


//ProfilingData_t* profilingData = new ProfilingData_t();

static void
profilingControlInit(int numRanges,int kernelPerRange,int numPassPerSess,string outputPath,string inputMetric,int autoRange,int kernelReplay) {
    //    ProfilerControl = {};
    ProfilerControl.initialized = 0;

    //    ProfilerControl.initialized = 0;
    ProfilerControl.CounterDataFilePath = outputPath;
    ProfilerControl.subscriber = 0;
    //    CUpti_SubscriberHandle  subscriber;
    // Track per-context profiler API data in a shared map
    //    mutex ctx_data_mutex;
    //    unordered_map<CUcontext, ctxProfilerData> ctx_data;
    // List of metrics to collect

    ProfilerControl.maxRangeNameLength=64;

    //    vector<string> metricNames;
    int numMetric = 0;
    cout << "Aim Metric: " << endl;
    // .c_str()
    const char* tmpInputMetric =inputMetric.c_str();
    char* gotmpInputMetric = const_cast<char*>(tmpInputMetric);
    char * tok = strtok(gotmpInputMetric, " ;,");
    do{
        cout << tok << endl;
        ProfilerControl.metricNames.push_back(string(tok));
        tok = strtok(NULL, " ;,");
        numMetric = numMetric + 1;
    } while (tok != NULL);
    unsigned long metricSize = ProfilerControl.metricNames.size();
    if((numMetric<1)&&(metricSize<1)){
        ProfilerControl.metricNames.push_back("sm__cycles_elapsed.avg");
        ProfilerControl.metricNames.push_back("smsp__sass_thread_inst_executed_op_dadd_pred_on.avg");
        ProfilerControl.metricNames.push_back("smsp__sass_thread_inst_executed_op_dfma_pred_on.avg");
    }
    ProfilerControl.auto_range = autoRange;
    ProfilerControl.kernel_replay = kernelReplay;

    if (ProfilerControl.auto_range>0){
        ProfilerControl.kernelPerRange = 1;
    }else{
        ProfilerControl.kernelPerRange = kernelPerRange;
    }

    if (ProfilerControl.kernelPerRange<=0){
        ProfilerControl.kernelPerRange = 1;
    }

    if (numRanges < 1){
        ProfilerControl.numRanges = 1;
    }else{
        ProfilerControl.numRanges = numRanges;
    }

    //    ProfilerControl.kernelPerRange = kernelPerRange;

    ProfilerControl.numPassPerSess = numPassPerSess;
}

// Print session profile data
static void print_profile_data(ctxProfilerData &ctx_data)
{
    cout << endl << "Context " << ctx_data.ctx << ", device " << ctx_data.dev_id << " (" << ctx_data.chipName << ") session " << ctx_data.iterations << ":" << endl;
    PrintMetricValues(ctx_data.chipName, ctx_data.counterDataImage, ProfilerControl.metricNames, ctx_data.counterAvailabilityImage.data());
    cout << endl;
}

// Print kernel trace data
//ctxProfilerData &ctx_data
//static void print_trace_data(ctxProfilerData &ctx_data)
//{
//
//    for(unordered_map<int,CallbackKernelTimestamp_t>::iterator it=ctx_data.kerneltime_data.begin();it!=ctx_data.kerneltime_data.end();it++){
//            CallbackKernelTimestamp_t* tmp_kernel_info = &it->second;
//            int rangeId = it->first;
//            // string kernelName;
//            // unsigned long long startTimestamp;
//            // unsigned long long endTimestamp;
//            // unsigned long long kernel_duration;
//            cout << endl << "Context ID: " << std::to_string(tmp_kernel_info->contextId) <<" Kernel: "<<tmp_kernel_info->kernelName<< " Index: "<<std::to_string(rangeId)<<" start: " << std::to_string(tmp_kernel_info->startTimestamp) << " end: "<< std::to_string(tmp_kernel_info->endTimestamp)<<" duration: "<<std::to_string(tmp_kernel_info->kernel_duration)<< endl;
//        }
//
//}






// Initialize state
void initialize_state()
{
//    ProfilerControl.initialized = 0;

    if (ProfilerControl.initialized == 0)
    {
        // CUPTI Profiler API initialization
        CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

        // NVPW required initialization
        NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
        NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

        ProfilerControl.initialized = 1;
    }
}

//std::vector<uint8_t>& counterDataImagePrefix,
//    std::vector<uint8_t>& counterDataScratchBuffer,
//    std::vector<uint8_t>& counterDataImage
void createCounterDataImage(int numRanges,int maxRangeNameLength,ctxProfilerData &ctx_data)
{
    // Record counterDataPrefixImage info and other options for sizing the counterDataImage
    ctx_data.counterDataImageOptions.pCounterDataPrefix = ctx_data.counterDataPrefixImage.data();
    ctx_data.counterDataImageOptions.counterDataPrefixSize = ctx_data.counterDataPrefixImage.size();
    ctx_data.counterDataImageOptions.maxNumRanges = numRanges;
    ctx_data.counterDataImageOptions.maxNumRangeTreeNodes = numRanges;
    ctx_data.counterDataImageOptions.maxRangeNameLength = maxRangeNameLength;

    // Calculate size of counterDataImage based on counterDataPrefixImage and options
    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = { CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE };
    calculateSizeParams.pOptions = &(ctx_data.counterDataImageOptions);
    calculateSizeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));
    // Create counterDataImage
    ctx_data.counterDataImage.resize(calculateSizeParams.counterDataImageSize);

    // Initialize counterDataImage inside start_session
    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
    initializeParams.pOptions = &(ctx_data.counterDataImageOptions);
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.counterDataImageSize = ctx_data.counterDataImage.size();
    initializeParams.pCounterDataImage = ctx_data.counterDataImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    // Calculate scratchBuffer size based on counterDataImage size and counterDataImage
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = { CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE };
    scratchBufferSizeParams.counterDataImageSize = ctx_data.counterDataImage.size();
    scratchBufferSizeParams.pCounterDataImage = ctx_data.counterDataImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));
    // Create counterDataScratchBuffer
    ctx_data.counterDataScratchBufferImage.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

     // Initialize counterDataScratchBuffer
    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = { CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE };
    initScratchBufferParams.counterDataImageSize = ctx_data.counterDataImage.size();
    initScratchBufferParams.pCounterDataImage = ctx_data.counterDataImage.data();
    initScratchBufferParams.counterDataScratchBufferSize = ctx_data.counterDataScratchBufferImage.size();
    initScratchBufferParams.pCounterDataScratchBuffer = ctx_data.counterDataScratchBufferImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

}

// Initialize profiler for a context
void initialize_ctx_data(ctxProfilerData &ctx_data)
{
    initialize_state();

    // Get size of counterAvailabilityImage - in first pass, GetCounterAvailability return size needed for data
    CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = { CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE };
    getCounterAvailabilityParams.ctx = ctx_data.ctx;
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    // Allocate sized counterAvailabilityImage
    ctx_data.counterAvailabilityImage.resize(getCounterAvailabilityParams.counterAvailabilityImageSize);

    // Initialize counterAvailabilityImage
    getCounterAvailabilityParams.pCounterAvailabilityImage = ctx_data.counterAvailabilityImage.data();
    CUPTI_API_CALL(cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

    // Fill in configImage - can be run on host or target
    if (!GetConfigImage(ctx_data.chipName, ProfilerControl.metricNames, ctx_data.configImage, ctx_data.counterAvailabilityImage.data()))
    {
        cerr << "Failed to create configImage for context " << ctx_data.ctx << endl;
        exit(EXIT_FAILURE);
    }

    // Fill in counterDataPrefixImage - can be run on host or target
    if (!GetCounterDataPrefixImage(ctx_data.chipName, ProfilerControl.metricNames, ctx_data.counterDataPrefixImage, ctx_data.counterAvailabilityImage.data()))
    {
        cerr << "Failed to create counterDataPrefixImage for context " << ctx_data.ctx << endl;
        exit(EXIT_FAILURE);
    }
    createCounterDataImage(ProfilerControl.numRanges,ProfilerControl.maxRangeNameLength,ctx_data);
}

bool beginSession(ctxProfilerData &ctx_data)
{
    bool out_res = false;
    if (ctx_data.session_started == true){
        return true;
    }else{
        CUpti_Profiler_BeginSession_Params beginSessionParams = { CUpti_Profiler_BeginSession_Params_STRUCT_SIZE };
        beginSessionParams.ctx = ctx_data.ctx;
        beginSessionParams.counterDataImageSize = ctx_data.counterDataImage.size();
        beginSessionParams.pCounterDataImage = ctx_data.counterDataImage.data();
        beginSessionParams.counterDataScratchBufferSize = ctx_data.counterDataScratchBufferImage.size();
        beginSessionParams.pCounterDataScratchBuffer = ctx_data.counterDataScratchBufferImage.data();
        if (ProfilerControl.auto_range>0){
            beginSessionParams.range = CUPTI_AutoRange;
            if (ProfilerControl.kernel_replay > 0){
                beginSessionParams.replayMode = CUPTI_KernelReplay;
            }else{
                beginSessionParams.replayMode = CUPTI_UserReplay;
            }
        }else{
            beginSessionParams.range = CUPTI_UserRange;
            beginSessionParams.replayMode = CUPTI_UserReplay;
        }

        beginSessionParams.maxRangesPerPass = ProfilerControl.numRanges;
        beginSessionParams.maxLaunchesPerPass = ProfilerControl.numRanges;
        CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

        ctx_data.session_started = true;
    }
    out_res = ctx_data.session_started;
    return out_res;
}

void setConfig(ctxProfilerData &ctx_data)
{
    CUpti_Profiler_SetConfig_Params setConfigParams = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
    setConfigParams.pConfig = ctx_data.configImage.data();
    setConfigParams.configSize = ctx_data.configImage.size();
    setConfigParams.passIndex = 0; // Only set for Application Replay mode
    setConfigParams.minNestingLevel = 1;
    setConfigParams.numNestingLevels = 1;
    setConfigParams.targetNestingLevel = 1;
    CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));

//    CUpti_Profiler_SetConfig_Params setConfigParams = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
//    setConfigParams.pConfig =
//    setConfigParams.configSize =

}

//if (pProfilingData->profilerReplayMode == CUPTI_UserReplay)

void enableProfiling(ctxProfilerData &ctx_data,bool start_pass,bool start_range,string rangeName)
{
    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = { CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE };
    enableProfilingParams.ctx = ctx_data.ctx;
    //    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
    if (ProfilerControl.kernel_replay > 0)
    {
        if (ctx_data.profile_enabled==false){
            CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
            ctx_data.profile_enabled = true;
            ctx_data.pass_started = true;
            ctx_data.range_started = true;
            cout<<endl<<"START TO ENABLE PROFILING"<<endl;
        }else{
            cout<<endl<<"PROFILING Have Enabled"<<endl;
        }



    }
    else
    {
        if((start_pass)&&(ctx_data.pass_started==false)){
            CUpti_Profiler_BeginPass_Params beginPassParams = { CUpti_Profiler_BeginPass_Params_STRUCT_SIZE };
            beginPassParams.ctx = ctx_data.ctx;
            CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));
            ctx_data.pass_started = true;
        }
        if (ctx_data.profile_enabled==false){
            CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
            ctx_data.profile_enabled = true;
        }

        if ((start_range)&&(ctx_data.range_started==false)){
            CUpti_Profiler_PushRange_Params pushRangeParams = {CUpti_Profiler_PushRange_Params_STRUCT_SIZE};
            pushRangeParams.pRangeName = rangeName.c_str();
            pushRangeParams.ctx = ctx_data.ctx;
            CUPTI_API_CALL(cuptiProfilerPushRange(&pushRangeParams));
            ctx_data.range_started = true;
        }
    }
}

void disableProfiling(ctxProfilerData &ctx_data,bool end_pass,bool end_range,string rangeName)
{
    if ((ProfilerControl.kernel_replay > 0)){
        if(ctx_data.profile_enabled==true){
            CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };
            disableProfilingParams.ctx = ctx_data.ctx;
            CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
            cout<<endl<<"TO DISABLE PROFILING"<<endl;
            ctx_data.profile_enabled = false;
            ctx_data.pass_started = false;
            ctx_data.range_started = false;
        }else{
            cout<<endl<<"PROFILING Have DISABLED"<<endl;
        }


        ctx_data.allPassesSubmitted = true;

    }else{
        if ((end_range)&&(ctx_data.range_started)){
            CUpti_Profiler_PopRange_Params popRangeParams = {CUpti_Profiler_PopRange_Params_STRUCT_SIZE};
            popRangeParams.ctx = ctx_data.ctx;
            CUPTI_API_CALL(cuptiProfilerPopRange(&popRangeParams));

            ctx_data.range_started = false;
        }

        if(ctx_data.profile_enabled==true){
            CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };
            disableProfilingParams.ctx = ctx_data.ctx;
            CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
            ctx_data.profile_enabled = false;
        }


        if ((end_pass)){
            if(ctx_data.pass_started){
                CUpti_Profiler_EndPass_Params endPassParams = { CUpti_Profiler_EndPass_Params_STRUCT_SIZE };
                endPassParams.ctx = ctx_data.ctx;
                CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));
                ctx_data.allPassesSubmitted = (endPassParams.allPassesSubmitted == 1) ? true : false;
                ctx_data.pass_started = false;
            }else{
                ctx_data.allPassesSubmitted = true;
            }
        }else{
            ctx_data.allPassesSubmitted = false;
        }

    }

    if (ctx_data.allPassesSubmitted)
    {
        CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = { CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE };
        CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));
    }
}

// Start a session
void start_session(ctxProfilerData &ctx_data,string rangeName)
{
    beginSession(ctx_data);
    setConfig(ctx_data);

    //    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = { CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE };
    //    enableProfilingParams.ctx = ctx_data.ctx;
    //    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
    enableProfiling(ctx_data,true,true,rangeName);
   //    enableProfiling()

    ctx_data.iterations++;
}

void unsetConfig(ctxProfilerData &ctx_data)
{
    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = { CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE };
    unsetConfigParams.ctx = ctx_data.ctx;

    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));

//    CUpti_Profiler_SetConfig_Params setConfigParams = { CUpti_Profiler_SetConfig_Params_STRUCT_SIZE };
//    setConfigParams.pConfig =
//    setConfigParams.configSize =

}

bool endSession(ctxProfilerData &ctx_data)
{
    bool out_res = false;
    if (ctx_data.session_started == false){
        out_res = true;
        return true;
    }else{
        //    CUpti_Profiler_BeginSession_Params beginSessionParams = { CUpti_Profiler_BeginSession_Params_STRUCT_SIZE };

        CUpti_Profiler_EndSession_Params endSessionParams = { CUpti_Profiler_EndSession_Params_STRUCT_SIZE };
        endSessionParams.ctx = ctx_data.ctx;
        CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));

        ctx_data.session_started = false;
        return true;
    }

}

// Initialize state
void finalize_state()
{
//    ProfilerControl.initialized = 0;

    if (ProfilerControl.initialized == 1)
    {
        // CUPTI Profiler API initialization
        CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
        CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));

        // NVPW required initialization
//        NVPW_InitializeHost_Params initializeHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE };
//        NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

        ProfilerControl.initialized = 0;
        ProfilerControl.nowContextId = 1;
    }
}

//void stopProfiling(ctxProfilerData &ctx_data)
//{
//
//    CUpti_Profiler_EndSession_Params endSessionParams = { CUpti_Profiler_EndSession_Params_STRUCT_SIZE };
//
//
//    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
//    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
//
//
//    // Dump counterDataImage and counterDataScratchBuffer in file.
//    WriteBinaryFile(pProfilingData->CounterDataFileName.c_str(), pProfilingData->counterDataImage);
//    WriteBinaryFile(pProfilingData->CounterDataSBFileName.c_str(), pProfilingData->counterDataScratchBuffer);
//}

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

bool store_profiling_results(ctxProfilerData &ctx_data){
    bool mk_path_res = createDirectory(ProfilerControl.CounterDataFilePath);

    if(mk_path_res){
        int session_iter = ctx_data.iterations;
        uint32_t context_id = 1;
        if (ProfilerControl.ctx_id.count(ctx_data.ctx)){
            context_id = ProfilerControl.ctx_id[ctx_data.ctx];
        }else{
            context_id = 1;
        }

        // device " << ctx_data.dev_id << " (" << ctx_data.dev_prop.name <<
        int dev_id = ctx_data.dev_id;
        string chipName = ctx_data.chipName;
        //        +"_"+chipName
        string base_file_name = ProfilerControl.CounterDataFilePath + "/" + "profile_"+std::to_string(startTimestamp)+"_ctx_"+std::to_string(context_id)+"_sess_"+std::to_string(session_iter)+"_dev_"+std::to_string(dev_id);
        string CounterDataFileName = base_file_name+".counterdata";
        string CounterDataSBFileName = base_file_name+".counterdataSB";

        WriteBinaryFile(CounterDataFileName.c_str(), ctx_data.counterDataImage);
        std::cout <<"Profile Data written successfully!" << std::endl;
        WriteBinaryFile(CounterDataSBFileName.c_str(), ctx_data.counterDataScratchBufferImage);
        std::cout <<"Profile Data Buffer written successfully!" << std::endl;

        return true;
    }else{
        std::cerr<<"Path Not Exists: Unable to create file for writing."<<std::endl;
        return false;
    }



}


//bool store_tracing_results(ctxProfilerData &ctx_data){
//    mk_path_res = createDirectory(ProfilerControl.CounterDataFilePath);
//
//    if(mk_path_res){
//        //        int session_iter = ctx_data.iterations;
//        //        uint32_t context_id = 1;
//        //        if (ProfilerControl.ctx_id.count(ctx_data.ctx)){
//        //            context_id = ProfilerControl.ctx_id[ctx_data.ctx];
//        //        }else{
//        //            context_id = 1;
//        //        }
//
//        string out_results = "Kernel,context,globalId,start,end,duration\n";
//        int session_iter = ctx_data.iterations;
//        for(unordered_map<int,CallbackKernelTimestamp_t>::iterator it=ctx_data.kerneltime_data.begin();it!=ctx_data.kerneltime_data.end();it++){
//            CallbackKernelTimestamp_t* tmp_kernel_info = &it->second;
//            int globalId = it->first;
//            // string kernelName;
//            // unsigned long long startTimestamp;
//            // unsigned long long endTimestamp;
//            // unsigned long long kernel_duration;
//            tmp_output_res = tmp_kernel_info->kernelName +","+std::to_string(tmp_kernel_info->contextId)+","+std::to_string(globalId)+"," + std::to_string(tmp_kernel_info->startTimestamp) +","+ std::to_string(tmp_kernel_info->endTimestamp) + "," + std::to_string(tmp_kernel_info->kernel_duration)+"\n";
//            out_results = out_results + tmp_output_res;
//        }
//        //        out_results.
//
//
//
//        // device " << ctx_data.dev_id << " (" << ctx_data.dev_prop.name <<
//        //        int dev_id = ctx_data.dev_id;
//        //        string chipName = ctx_data.dev_prop.name;
//        //        +"_"+chipName
//        string base_file_name = ProfilerControl.CounterDataFilePath + "/" + "kernel_trace_"+std::to_string(startTimestamp)+"_sess_"+std::to_string(session_iter);
//        out_file_name = base_file_name+".csv";
//
//        std::ofstream outFile(out_file_name);
//
//        if (outFile.is_open()){
//            outFile << out_results;
//            outFile.close();
//            std::cout <<"Trace Data written successfully!" << std::endl;
//        }else{
//            std::cerr<<"Unable to open file for writing."<<std::endl;
//            return false;
//        }
//        //        CounterDataSBFileName = base_file_name+".counterdataSB"
//        //
//        //        WriteBinaryFile(CounterDataFileName.c_str(), ctx_data.counterDataImage);
//        //        WriteBinaryFile(CounterDataSBFileName.c_str(), ctx_data.counterDataScratchBufferImage);
//
//        return true;
//    }else{
//        std::cerr<<"Path Not Exists: Unable to create file for writing."<<std::endl;
//        return false;
//    }
//
//
//}

void cleanCounterDataImage(ctxProfilerData &ctx_data){
    // Clear counterDataImage (otherwise it maintains previous records when it is reused)
    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = { CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE };
    initializeParams.pOptions = &(ctx_data.counterDataImageOptions);
    initializeParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initializeParams.counterDataImageSize = ctx_data.counterDataImage.size();
    initializeParams.pCounterDataImage = ctx_data.counterDataImage.data();
    CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));
}

//void cleanTraceData(ctxProfilerData &ctx_data){
//    ctx_data.kerneltime_data.clear();
//}

// End a session during execution
void end_session(ctxProfilerData &ctx_data)
{
    //    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = { CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE };
    //    disableProfilingParams.ctx = ctx_data.ctx;
    //    CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));

    //    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = { CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE };
    //    unsetConfigParams.ctx = ctx_data.ctx;
    // disableProfiling(ctxProfilerData &ctx_data,bool end_pass,bool end_range,string rangeName)
    disableProfiling(ctx_data,true,true,"");
    unsetConfig(ctx_data);

    endSession(ctx_data);



    //print_trace_data();
    //store_tracing_results(ctx_data);
    print_profile_data(ctx_data);
    store_profiling_results(ctx_data);
    cleanCounterDataImage(ctx_data);
    //cleanTraceData(ctx_data);

}

void create_ctx_profile_env(ctxProfilerData &data,CUcontext &ctx){
    // Initialize profiler API and test device compatibility
    initialize_state();
    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
    params.cuDevice = data.dev_id;
    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    // If valid for profiling, set up profiler and save to shared structure
    ProfilerControl.ctx_data_mutex.lock();
    if (params.isSupported == CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        // Update shared structures
        data.curRanges = 0;
        ProfilerControl.ctx_data[ctx] = data;
        initialize_ctx_data(ProfilerControl.ctx_data[ctx]);

    }else
    {
        if (ProfilerControl.ctx_data.count(ctx))
        {
            // Update shared structures
            ProfilerControl.ctx_data.erase(ctx);
        }

        cerr << "Callback profiling: Unable to profile context on device " << data.dev_id << endl;

        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            cerr << "\tdevice architecture is not supported" << endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            cerr << "\tdevice sli configuration is not supported" << endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            cerr << "\tdevice vgpu configuration is not supported" << endl;
        }
        else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
        {
            cerr << "\tdevice vgpu configuration disabled profiling support" << endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            cerr << "\tdevice confidential compute configuration is not supported" << endl;
        }

        if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
        }
    }
    ProfilerControl.ctx_data_mutex.unlock();
}



// Callback handler
void CallbackHandler(void * userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void const * cbdata)
{
    static int initialized = 0;

    CUptiResult res;
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API)
    {
        // For a driver call to launch a kernel:
        if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)
        {
            cout<<endl<<"FIND DRIVER KERNEL LAUNCH"<<endl;
            CUpti_CallbackData const * data = static_cast<CUpti_CallbackData const *>(cbdata);
            CUcontext ctx = data->context;
            uint32_t ctxId = data->contextUid;

            if (ProfilerControl.ctx_id.count(ctx) <= 0){
                ProfilerControl.ctx_data_mutex.lock();
                ProfilerControl.ctx_id[ctx] = ctxId;
                ProfilerControl.ctx_data_mutex.unlock();
            }
            string kernelName;
            kernelName = data->symbolName;

            int numRanges = ProfilerControl.numRanges;
            int kernelPerRange = 1;
            if (ProfilerControl.auto_range > 0){
                kernelPerRange = 1;
            }else{
                kernelPerRange = ProfilerControl.kernelPerRange;
            }

            int numPassPerSess = ProfilerControl.numPassPerSess;

            int maxNumRangeSess = numRanges*kernelPerRange*numPassPerSess;
            int maxNumRangePass = numRanges*kernelPerRange;



            // On entry, enable / update profiling as needed
            if (data->callbackSite == CUPTI_API_ENTER)
            {
                  //uint64_t kernelStartTimestamp;
                  //uint64_t kernelEndTimestamp;
                  //string kernelName;
                  //uint32_t contextId;
                  //uint64_t startTimestamp;
                  //uint64_t endTimestamp;
                  //uint64_t kernel_duration;
                  //CallbackKernelTimestamp_t kernel_trace_data = { };
                  // Collect timestamp for API start
                  //CUPTI_CALL(cuptiGetTimestamp(&kernelStartTimestamp));
                  //kernel_trace_data.contextId = ctx_id;
                  //kernel_trace_data.kernelName = kernelName;
                  //kernel_trace_data.startTimestamp = kernelStartTimestamp
                  ////traceData->startTimestamp = startTimestamp;
                  //ProfilerControl.ctx_data_mutex.lock();
                  //ProfilerControl.ctx_data[ctx].kerneltime_data[ProfilerControl.ctx_data[ctx].curRanges] = kernel_trace_data;
                  //ProfilerControl.ctx_data_mutex.unlock();


                // Check for this context in the configured contexts
                // If not configured, it isn't compatible with profiling
                if (ProfilerControl.ctx_data.count(ctx) <= 0){
                    // Configure handler for new context under lock
                    ctxProfilerData cpro_data = { };

                    cpro_data.ctx = ctx;
                    cpro_data.curRanges = 0;

                    RUNTIME_API_CALL(cudaGetDevice(&(cpro_data.dev_id)));

                    CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };
                    getChipNameParams.deviceIndex = cpro_data.dev_id;
                    CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
                    cpro_data.chipName = getChipNameParams.pChipName;
                    //  profilingData->chipName = getChipNameParams.pChipName;



                    create_ctx_profile_env(cpro_data,ctx);

                }
                ProfilerControl.ctx_data_mutex.lock();
                if (ProfilerControl.ctx_data.count(ctx) > 0)
                {

                    if (maxNumRangeSess >0){
                        if ((ProfilerControl.ctx_data[ctx].curRanges>0)){
                            if(((ProfilerControl.ctx_data[ctx].curRanges % maxNumRangePass)==0)&&(maxNumRangePass>1)){
                                //disableProfiling(ctxProfilerData &ctx_data,bool end_pass,bool end_range,string rangeName)
                                disableProfiling(ProfilerControl.ctx_data[ctx],true,true,"");
                            }else if(((ProfilerControl.ctx_data[ctx].curRanges % kernelPerRange)==0)&&(kernelPerRange>1)){
                                disableProfiling(ProfilerControl.ctx_data[ctx],false,true,"");
                            }
                            if ((ProfilerControl.ctx_data[ctx].curRanges % maxNumRangeSess)==0){
                                end_session(ProfilerControl.ctx_data[ctx]);
                            }
                        }
                        string range_name = kernelName + "_" + std::to_string(ProfilerControl.ctx_data[ctx].curRanges);
                        if ((ProfilerControl.ctx_data[ctx].curRanges % maxNumRangeSess)==0||(ProfilerControl.ctx_data[ctx].session_started==false)){
                            start_session(ProfilerControl.ctx_data[ctx],range_name);
                        }

                        if((ProfilerControl.ctx_data[ctx].curRanges % maxNumRangePass)==0){
                            //  enableProfiling(ctxProfilerData &ctx_data,bool start_pass,bool start_range,string rangeName)
                            enableProfiling(ProfilerControl.ctx_data[ctx],true,true,range_name);
                        }
                        else if((ProfilerControl.ctx_data[ctx].curRanges % kernelPerRange)==0){
                            enableProfiling(ProfilerControl.ctx_data[ctx],false,true,range_name);
                        }

                    }else{
                        if((ProfilerControl.ctx_data[ctx].curRanges>0)){
                            if(((ProfilerControl.ctx_data[ctx].curRanges % maxNumRangePass)==0)&&(maxNumRangePass>1)){
                                //disableProfiling(ctxProfilerData &ctx_data,bool end_pass,bool end_range,string rangeName)
                                disableProfiling(ProfilerControl.ctx_data[ctx],true,true,"");
                            }else if(((ProfilerControl.ctx_data[ctx].curRanges % kernelPerRange)==0)&&(kernelPerRange>1)){
                                disableProfiling(ProfilerControl.ctx_data[ctx],false,true,"");
                            }
                            //if ((ProfilerControl.ctx_data[ctx].curRanges % maxNumRangeSess)==0){
                            //    end_session(ProfilerControl.ctx_data[ctx]);
                            //}
                        }
                        string range_name = kernelName + "_" + std::to_string(ProfilerControl.ctx_data[ctx].curRanges);
                        if((ProfilerControl.ctx_data[ctx].curRanges==0)||(ProfilerControl.ctx_data[ctx].session_started==false)){
                              start_session(ProfilerControl.ctx_data[ctx],range_name);
                        }

                        if((ProfilerControl.ctx_data[ctx].curRanges % maxNumRangePass)==0){
                              //  enableProfiling(ctxProfilerData &ctx_data,bool start_pass,bool start_range,string rangeName)
                              enableProfiling(ProfilerControl.ctx_data[ctx],true,true,range_name);
                        }
                        else if((ProfilerControl.ctx_data[ctx].curRanges % kernelPerRange)==0){
                              enableProfiling(ProfilerControl.ctx_data[ctx],false,true,range_name);
                        }
                        //else{
                        //      enableProfiling(ProfilerControl.ctx_data[ctx],false,false,range_name)
                        //}
                    }
                    // Increment curRanges
                    ProfilerControl.ctx_data[ctx].curRanges++;
                }
                ProfilerControl.ctx_data_mutex.unlock();
            }
        }
    }
    else if (domain == CUPTI_CB_DOMAIN_RESOURCE)
    {
        // When a context is created, check to see whether the device is compatible with the Profiler API
        if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED)
        {
            CUpti_ResourceData const * res_data = static_cast<CUpti_ResourceData const *>(cbdata);
            CUcontext ctx = res_data->context;

            // Configure handler for new context under lock
            ctxProfilerData cpro_data = { };

            cpro_data.curRanges = 0;

            cpro_data.ctx = ctx;

            RUNTIME_API_CALL(cudaGetDevice(&(cpro_data.dev_id)));

            CUpti_Device_GetChipName_Params getChipNameParams = { CUpti_Device_GetChipName_Params_STRUCT_SIZE };

            getChipNameParams.deviceIndex = cpro_data.dev_id;

            CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));

            cpro_data.chipName = getChipNameParams.pChipName;

            create_ctx_profile_env(cpro_data,ctx);


        }
    }

    return;
}

// Clean up at end of execution
static void end_execution()
{
    CUPTI_API_CALL(cuptiGetLastError());
    ProfilerControl.ctx_data_mutex.lock();

    for (auto itr = ProfilerControl.ctx_data.begin(); itr != ProfilerControl.ctx_data.end(); ++itr)
    {
        ctxProfilerData &data = itr->second;

//        if (data.ran)
        if (data.session_started)
        {
            //            print_data(data);
            end_session(data);
            data.curRanges = 0;
        }
    }

    ProfilerControl.ctx_data_mutex.unlock();
}

// Register callbacks for several points in target application execution
void register_callbacks()
{
    // One subscriber is used to register multiple callback domains
//    CUpti_SubscriberHandle subscriber;
    CUPTI_API_CALL(cuptiSubscribe(&ProfilerControl.subscriber, (CUpti_CallbackFunc)CallbackHandler, NULL));
    // Runtime callback domain is needed for kernel launch callbacks
    CUPTI_API_CALL(cuptiEnableCallback(1, ProfilerControl.subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    // Resource callback domain is needed for context creation callbacks
    CUPTI_API_CALL(cuptiEnableCallback(1, ProfilerControl.subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED));

    // Register callback for application exit
    atexit(end_execution);
}

bool InitialCallbackProfiler(int numRanges=10,int kernelPerRange=1,int numPassPerSess=-1,string outputPath="profile_results",string inputMetric="smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",int autoRange=1,int kernelReplay=1){

    profilingControlInit(numRanges,kernelPerRange,numPassPerSess,outputPath,inputMetric,autoRange,kernelReplay);
    CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
    register_callbacks();

    return true;


}



bool FinalizeCallbackProfiler(){

    end_execution();
    finalize_state();
    CUPTI_CALL(cuptiFinalize());
    return true;

}

PYBIND11_MODULE(async_cupti_profile, m){
    m.doc() = "pybind11 for async cupti profiling";
    m.def("InitialCallbackProfiler", &InitialCallbackProfiler, "A function which starts a callback profiling procedure");
    m.def("FinalizeCallbackProfiler",&FinalizeCallbackProfiler,"A function which destories a callback profiling procedure");
    // m.def("inadd", &inadd, "cin and cout");
}