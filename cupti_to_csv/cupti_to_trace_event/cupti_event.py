from .cupti_strings import RUNTIME_CBID_NAMES, DRIVER_CBID_NAMES

import re


class CUPTItoTraceEvent:
    StartTS = None

    def __init__(self, line, pattern):
        match = re.match(pattern, line)
        assert match, line

        self.data = match.groupdict()
        self.evaluate_numeric_values()

    def evaluate_numeric_values(self):
        for key, value in self.data.items():
            if value.isdigit():
                self.data[key] = eval(value)

    def ns_to_us(self, ns):
        return ns * 1e-3

    def start_ts(self):
        assert CUPTItoTraceEvent.StartTS is not None
        return CUPTItoTraceEvent.StartTS

    def convert(self):
        raise NotImplementedError(
            "the subclass must implement the convert method")


class MemCopyEvent(CUPTItoTraceEvent): #类型后的空格问题
    def __init__(self, line):
        super().__init__(
            line, r"(?P<type>\w+)\s(?P<direction>\w+)\[\s(?P<start>\d+)\s-\s(?P<end>\d+)\s\]\sdevice\s(?P<device>\d+),\scontext\s(?P<context>\d+),\sstream\s(?P<stream>\d+),\ssize\s(?P<size>\d+),\scorrelation\s(?P<correlation>\d+)")

    def convert(self):
        return {
            #"ph": "X",
            "cat": "gpu_memcpy",
            "name" : f"Memcpy {self.data['direction']}",
            "start": self.ns_to_us(self.data["start"] + self.start_ts()),
            "end": self.ns_to_us(self.data["end"] + self.start_ts()),
            "dur":round(self.ns_to_us(self.data["end"] - self.data["start"]),3),
            #"args": {
                # tmp
                #"External id": 0,
                "device": self.data["device"],
                "context": self.data["context"],
                "stream": self.data["stream"],
                "correlation": self.data["correlation"],
                "bytes": self.data["size"],
                # "memory bandwidth (GB/s)":
            #}
        }


class MemSetEvent(CUPTItoTraceEvent): #类型后的空格问题
    def __init__(self, line):
        super().__init__(
            line, r"(?P<type>\w+) value=(?P<value>\d+)\[ (?P<start>\d+) - (?P<end>\d+) \] device (?P<device>\d+), context (?P<context>\d+), stream (?P<stream>\d+), size (?P<size>\d+), correlation (?P<correlation>\d+)")

    def convert(self):
        return {
            #"ph": "X",
            "cat": "gpu_memset",
            "name": "Memset",
            "start": self.ns_to_us(self.data["start"] + self.start_ts()),
            "end": self.ns_to_us(self.data["end"] + self.start_ts()),
            "dur":round(self.ns_to_us(self.data["end"] - self.data["start"]),3),
            #"args": {
                # tmp
                "External id": 0,
                "device": self.data["device"],
                "context": self.data["context"],
                "stream": self.data["stream"],
                "correlation": self.data["correlation"],
                "bytes": self.data["size"],
                # "memory bandwidth (GB/s)":
            #}
        }


class CudaRuntimeEvent(CUPTItoTraceEvent):
    def __init__(self, line):
        super().__init__(line,
                         r"(?P<type>\w+) cbid=(?P<cbid>\d+) \[ (?P<start>\d+) - (?P<end>\d+) \] process (?P<pid>\d+), thread (?P<tid>\d+), correlation (?P<correlation>\d+)")

    def convert(self):
        return {
            #"ph": "X",
            "cat": "cuda_runtime",
            "name": RUNTIME_CBID_NAMES[self.data["cbid"]],
            "start": self.ns_to_us(self.data["start"] + self.start_ts()),
            "end": self.ns_to_us(self.data["end"] + self.start_ts()),
            "dur":round(self.ns_to_us(self.data["end"] - self.data["start"]),3),
            "pid": self.data["pid"],
            # tmp
            "tid": self.data["tid"],
            #"args": {
                # tmp
                #"External id": 0,
                "cbid": self.data["cbid"],
                "correlation": self.data["correlation"]
            #}
        }


class CudaDriverEvent(CUPTItoTraceEvent):
    def __init__(self, line):
        super().__init__(line,
                         r"(?P<type>\w+) cbid=(?P<cbid>\d+) \[ (?P<start>\d+) - (?P<end>\d+) \] process (?P<pid>\d+), thread (?P<tid>\d+), correlation (?P<correlation>\d+)")

    def convert(self):
        # if self.data["cbid"] >= len(DRIVER_CBID_NAMES):
        #     print(self.data["cbid"])
        #     print(len(DRIVER_CBID_NAMES))
        # elif DRIVER_CBID_NAMES[self.data["cbid"]] != "cuLaunchKernel":
        #     # currently only cuLaunchKernel is expected
        #     return None

        return {
            #"ph": "X",
            "cat": "cuda_driver",
            "name": DRIVER_CBID_NAMES[self.data["cbid"]],
            "start": self.ns_to_us(self.data["start"] + self.start_ts()),
            "end": self.ns_to_us(self.data["end"] + self.start_ts()),
            "dur":round(self.ns_to_us(self.data["end"] - self.data["start"]),3),
            "pid": self.data["pid"],
            # tmp
            "tid": self.data["tid"],
            #"args": {
                # tmp
                #"External id": 0,
                "cbid": self.data["cbid"],
                "correlation": self.data["correlation"]
            #}
        }


class KernelEvent(CUPTItoTraceEvent): #类型后的空格问题
    def __init__(self, line):
        super().__init__(line, r'(?P<type>KERNEL)"(?P<name>[^"]+)" \[ (?P<start>\d+) - (?P<end>\d+) \] device (?P<device>\d+), context (?P<context>\d+), stream (?P<stream>\d+), correlation (?P<correlation>\d+), grid \[(?P<grid>[^\]]+)\], block \[(?P<block>[^\]]+)\], shared memory \(static (?P<static_shared_memory>\d+), dynamic (?P<dynamic_shared_memory>\d+)\)')
   
    def convert(self):
        return {
            #"ph": "X",
            "cat": "kernel",
            "name": self.data["name"],
            #特殊处理一下kernel 时间
            "start": self.ns_to_us(self.data["start"]),
            "end": self.ns_to_us(self.data["end"]),
            "dur":round(self.ns_to_us(self.data["end"] - self.data["start"]),3),
            #"args": {
                # tmp
                #"External id": 0,
                # tmp
                #"queued": 0,
                "device": self.data["device"],
                "context": self.data["context"],
                "stream": self.data["stream"],
                "correlation": self.data["correlation"],
                #"registers per thread": self.data["register_per_thread"],
                "shared memory": self.data["static_shared_memory"] + self.data["dynamic_shared_memory"],
                # "blocks per SM": 0.012821,
                # "warps per SM": 0.051282,
                "grid": self.data["grid"],
                "block": self.data["block"],
                # "est. achieved occupancy %": 0
            #}
        }
    
class ConcKernelEvent(CUPTItoTraceEvent): #类型后的空格问题
    def __init__(self, line):
        super().__init__(line, r'(?P<type>CONC KERNEL)"(?P<name>[^"]+)" \[ (?P<start>\d+) - (?P<end>\d+) \] device (?P<device>\d+), context (?P<context>\d+), stream (?P<stream>\d+), correlation (?P<correlation>\d+), grid \[(?P<grid>[^\]]+)\], block \[(?P<block>[^\]]+)\], shared memory \(static (?P<static_shared_memory>\d+), dynamic (?P<dynamic_shared_memory>\d+)\)')
    def convert(self):
        return {
            #"ph": "X",
            "cat": "kernel",
            "name": self.data["name"],
            #特殊处理一下kernel 时间
            "start": self.ns_to_us(self.data["start"]),
            "end": self.ns_to_us(self.data["end"]),  
            "dur":round(self.ns_to_us(self.data["end"] - self.data["start"]),3),
            #"args": {
                # tmp
                #"External id": 0,
                # tmp
                #"queued": 0,
                "device": self.data["device"],
                "context": self.data["context"],
                "stream": self.data["stream"],
                "correlation": self.data["correlation"],
                "shared memory": self.data["static_shared_memory"] + self.data["dynamic_shared_memory"],
                "grid": self.data["grid"],
                "block": self.data["block"],
            #}
        }


class StartTSEvent(CUPTItoTraceEvent):
    def __init__(self, line):
        super().__init__(line, r"CUPTI_START_TIMESTAMP\s*=\s*(?P<ts>\d+)")

    def convert(self):
        CUPTItoTraceEvent.StartTS = self.data["ts"]
        return {
            "name": "Record Window Start",
            "ph": "i",
            "s": "g",
            "pid": "Traces",
            "tid": "Traces",
            "ts": self.ns_to_us(self.data["ts"])
        }
