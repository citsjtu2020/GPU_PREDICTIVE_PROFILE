#include <Eval.h>
#include <Parser.h>
#include <Utils.h>
#include <FileOp.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>
#include <iostream>
//#include <string>
//#include <sstream>
#include <iomanip>
#include <ScopeExit.h>

#include <unordered_map>
using ::std::unordered_map;

namespace NV {
    namespace Metric {
        namespace Eval {
            std::string GetHwUnit(const std::string& metricName)
            {
                return metricName.substr(0, metricName.find("__", 0));
            }

            bool GetMetricGpuValue(std::string chipName,
                                    const std::vector<uint8_t>& counterDataImage,
                                    const std::vector<std::string>& metricNames,
                                    std::vector<MetricNameValue>& metricNameValueMap,
                                    const uint8_t* pCounterAvailabilityImage)
            {
                if (!counterDataImage.size())
                {
                    std::cout << "Counter Data Image is empty!\n";
                    return false;
                }

                NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
                calculateScratchBufferSizeParam.pChipName = chipName.c_str();
                calculateScratchBufferSizeParam.pCounterAvailabilityImage = pCounterAvailabilityImage;
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calculateScratchBufferSizeParam));

                std::vector<uint8_t> scratchBuffer(calculateScratchBufferSizeParam.scratchBufferSize);
                NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
                metricEvaluatorInitializeParams.scratchBufferSize = scratchBuffer.size();
                metricEvaluatorInitializeParams.pScratchBuffer = scratchBuffer.data();
                metricEvaluatorInitializeParams.pChipName = chipName.c_str();
                metricEvaluatorInitializeParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvaluatorInitializeParams));
                NVPW_MetricsEvaluator* metricEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;

                NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
                getNumRangesParams.pCounterDataImage = counterDataImage.data();
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterData_GetNumRanges(&getNumRangesParams));

                bool isolated = true;
                bool keepInstances = true;
                for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex)
                {
                    std::string reqName;
                    NV::Metric::Parser::ParseMetricNameString(metricNames[metricIndex], &reqName, &isolated, &keepInstances);
                    NVPW_MetricEvalRequest metricEvalRequest;
                    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertMetricToEvalRequest = {NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
                    convertMetricToEvalRequest.pMetricsEvaluator = metricEvaluator;
                    convertMetricToEvalRequest.pMetricName = reqName.c_str();
                    convertMetricToEvalRequest.pMetricEvalRequest = &metricEvalRequest;
                    convertMetricToEvalRequest.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
                    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&convertMetricToEvalRequest));
                    
                    MetricNameValue metricNameValue;
                    metricNameValue.numRanges = getNumRangesParams.numRanges;
                    metricNameValue.metricName = metricNames[metricIndex];
                    for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex)
                    {
                        NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = { NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
                        getRangeDescParams.pCounterDataImage = counterDataImage.data();
                        getRangeDescParams.rangeIndex = rangeIndex;
                        RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));
                        std::vector<const char*> descriptionPtrs(getRangeDescParams.numDescriptions);
                        getRangeDescParams.ppDescriptions = descriptionPtrs.data();
                        RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

                        std::string rangeName;
                        for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
                        {
                            if (descriptionIndex)
                            {
                                rangeName += "/";
                            }
                            rangeName += descriptionPtrs[descriptionIndex];
                        }

                        NVPW_MetricsEvaluator_SetDeviceAttributes_Params setDeviceAttribParams = { NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE };
                        setDeviceAttribParams.pMetricsEvaluator = metricEvaluator;
                        setDeviceAttribParams.pCounterDataImage = counterDataImage.data();
                        setDeviceAttribParams.counterDataImageSize = counterDataImage.size();
                        RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_SetDeviceAttributes(&setDeviceAttribParams));

                        double metricValue = 0.0;
                        NVPW_MetricsEvaluator_EvaluateToGpuValues_Params evaluateToGpuValuesParams = { NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE };
                        evaluateToGpuValuesParams.pMetricsEvaluator = metricEvaluator;
                        evaluateToGpuValuesParams.pMetricEvalRequests = &metricEvalRequest;
                        evaluateToGpuValuesParams.numMetricEvalRequests = 1;
                        evaluateToGpuValuesParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
                        evaluateToGpuValuesParams.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
                        evaluateToGpuValuesParams.pCounterDataImage = counterDataImage.data();
                        evaluateToGpuValuesParams.counterDataImageSize = counterDataImage.size();
                        evaluateToGpuValuesParams.rangeIndex = rangeIndex;
                        evaluateToGpuValuesParams.isolated = true;
                        evaluateToGpuValuesParams.pMetricValues = &metricValue;
                        RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_EvaluateToGpuValues(&evaluateToGpuValuesParams));
                        metricNameValue.rangeNameMetricValueMap.push_back(std::make_pair(rangeName, metricValue));
                    }
                    metricNameValueMap.push_back(metricNameValue);
                }
                
                NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = { NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE };
                metricEvaluatorDestroyParams.pMetricsEvaluator = metricEvaluator;
                RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams));
                return true;
            }

            bool PrintMetricValues( std::string chipName,
                                    const std::vector<uint8_t>& counterDataImage,
                                    const std::vector<std::string>& metricNames,
                                    int auto_range,
                                    unordered_map<int,std::string>& kernel_range_id_name,
                                    const uint8_t* pCounterAvailabilityImage
                                    )
            {
                if (!counterDataImage.size())
                {
                    std::cout << "Counter Data Image is empty!\n";
                    return false;
                }

                NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calculateScratchBufferSizeParam = {NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
                calculateScratchBufferSizeParam.pChipName = chipName.c_str();
                calculateScratchBufferSizeParam.pCounterAvailabilityImage = pCounterAvailabilityImage;
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calculateScratchBufferSizeParam));

                std::vector<uint8_t> scratchBuffer(calculateScratchBufferSizeParam.scratchBufferSize);
                NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams = {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
                metricEvaluatorInitializeParams.scratchBufferSize = scratchBuffer.size();
                metricEvaluatorInitializeParams.pScratchBuffer = scratchBuffer.data();
                metricEvaluatorInitializeParams.pChipName = chipName.c_str();
                metricEvaluatorInitializeParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
                metricEvaluatorInitializeParams.pCounterDataImage = counterDataImage.data();
                metricEvaluatorInitializeParams.counterDataImageSize = counterDataImage.size();
                RETURN_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_Initialize(&metricEvaluatorInitializeParams));
                NVPW_MetricsEvaluator* metricEvaluator = metricEvaluatorInitializeParams.pMetricsEvaluator;

                NVPW_CounterData_GetNumRanges_Params getNumRangesParams = { NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE };
                getNumRangesParams.pCounterDataImage = counterDataImage.data();
                RETURN_IF_NVPW_ERROR(false, NVPW_CounterData_GetNumRanges(&getNumRangesParams));


                if (auto_range > 0){
                    std::cout << "\n" << std::setw(40) << std::left << "Range"<<","<<"Kernel,"<< std::setw(80) << std::left        << "Metric Name,"
                          << "Metric Value" << std::endl;
                }else{
                    std::cout << "\n" << std::setw(40) << std::left << "Range Name,"
                          << std::setw(80) << std::left        << "Metric Name,"
                          << "Metric Value," << std::endl;
                }

                std::cout << std::setfill('-') << std::setw(160) << "" << std::setfill(' ') << std::endl;

                std::string reqName;
                bool isolated = true;
                bool keepInstances = true;
                for (std::string metricName : metricNames)
                {
                    NV::Metric::Parser::ParseMetricNameString(metricName, &reqName, &isolated, &keepInstances);
                    NVPW_MetricEvalRequest metricEvalRequest;
                    NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params convertMetricToEvalRequest = {NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
                    convertMetricToEvalRequest.pMetricsEvaluator = metricEvaluator;
                    convertMetricToEvalRequest.pMetricName = reqName.c_str();
                    convertMetricToEvalRequest.pMetricEvalRequest = &metricEvalRequest;
                    convertMetricToEvalRequest.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
                    RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&convertMetricToEvalRequest));

                    for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex)
                    {
                        NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = { NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE };
                        getRangeDescParams.pCounterDataImage = counterDataImage.data();
                        getRangeDescParams.rangeIndex = rangeIndex;
                        RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));
                        std::vector<const char*> descriptionPtrs(getRangeDescParams.numDescriptions);
                        getRangeDescParams.ppDescriptions = descriptionPtrs.data();
                        RETURN_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

                        std::string rangeName;
                        for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex)
                        {
                            if (descriptionIndex)
                            {
                                rangeName += "/";
                            }
                            rangeName += descriptionPtrs[descriptionIndex];
                        }

                        NVPW_MetricsEvaluator_SetDeviceAttributes_Params setDeviceAttribParams = { NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE };
                        setDeviceAttribParams.pMetricsEvaluator = metricEvaluator;
                        setDeviceAttribParams.pCounterDataImage = counterDataImage.data();
                        setDeviceAttribParams.counterDataImageSize = counterDataImage.size();
                        RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_SetDeviceAttributes(&setDeviceAttribParams));

                        double metricValue;
                        NVPW_MetricsEvaluator_EvaluateToGpuValues_Params evaluateToGpuValuesParams = { NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE };
                        evaluateToGpuValuesParams.pMetricsEvaluator = metricEvaluator;
                        evaluateToGpuValuesParams.pMetricEvalRequests = &metricEvalRequest;
                        evaluateToGpuValuesParams.numMetricEvalRequests = 1;
                        evaluateToGpuValuesParams.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
                        evaluateToGpuValuesParams.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
                        evaluateToGpuValuesParams.pCounterDataImage = counterDataImage.data();
                        evaluateToGpuValuesParams.counterDataImageSize = counterDataImage.size();
                        evaluateToGpuValuesParams.rangeIndex = rangeIndex;
                        evaluateToGpuValuesParams.isolated = true;
                        evaluateToGpuValuesParams.pMetricValues = &metricValue;
                        RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_EvaluateToGpuValues(&evaluateToGpuValuesParams));

                        int rangeInt = -1;
                        std::string kernel_name;
                        if (auto_range > 0){
                            rangeInt = stringToIntWithoutSpaces(rangeName);
                            if (kernel_range_id_name.count(rangeInt) > 0){
                                kernel_name = kernel_range_id_name[rangeInt];
                            }else{
                                kernel_name = "<unknown>";
                            }
                        }
                        if (rangeInt >= 0){
                            std::cout << std::setw(40) << std::left << std::to_string(rangeInt)<<","<< kernel_name<<"," << std::setw(80)
                                  << std::left << metricName <<","<< metricValue << std::endl;
                        }else{
                            std::cout << std::setw(40) << std::left << rangeName<<"," << std::setw(80)
                                  << std::left << metricName << ","<<metricValue << std::endl;
                        }

                    }
                }
                
                NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = { NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE };
                metricEvaluatorDestroyParams.pMetricsEvaluator = metricEvaluator;
                RETURN_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams));
                return true;
            }
        }
    }
}
