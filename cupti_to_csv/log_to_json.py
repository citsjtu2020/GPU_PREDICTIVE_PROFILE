from cupti_to_trace_event import CUPTItoTraceEventConverter

import os


path="cupti_to_csv/log"
# file_name="trace_no_profiler_con_res"


def setup_converter(input_path, output_path):
    converter = CUPTItoTraceEventConverter(input_path, output_path)
    return converter, output_path


def test_conversion(setup_converter):
    converter, output_path = setup_converter
    converter.convert()
    assert os.path.exists(output_path)


# def teardown_module(module):
#     output_path = "./examples/cupti.pt.trace.json"
#     if os.path.exists(output_path):
#         os.remove(output_path)


if __name__=="__main__":
    
    assert os.path.exists(path)
    log_files = [file_name[:-4] for file_name in os.listdir(path) if file_name.endswith(".log")]
    for file_name in log_files:
        input_path=os.path.join(path, file_name)+".log"
        output_path=os.path.join(path, file_name)+".json"
        test_conversion(setup_converter(input_path, output_path))
