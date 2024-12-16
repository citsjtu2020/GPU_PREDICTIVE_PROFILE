from .cupti_event import MemCopyEvent, MemSetEvent, CudaRuntimeEvent, CudaDriverEvent, KernelEvent, StartTSEvent, ConcKernelEvent

import json
import re


class CUPTItoTraceEventConverter:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def create_event(self, event_type, line):
        event_classes = {
            "MEMCPY": MemCopyEvent,
            "MEMSET": MemSetEvent,
            "RUNTIME": CudaRuntimeEvent,
            "DRIVER": CudaDriverEvent,
            "CONC": ConcKernelEvent,
            "KERNEL":KernelEvent,
            "CUPTI_START_TIMESTAMP": StartTSEvent
        }
        event_class = event_classes.get(event_type)
        if event_class:
            return event_class(line).convert()
        return None

    def convert_line(self, line):
        match = re.match(r'(?P<type>\w+)', line)
        if match:
            data = match.groupdict()
            event_type = data['type']
            return self.create_event(event_type, line)
        return None

    def convert(self):
        trace_events = []
        with open(self.input_file, 'r') as infile:
            for line in infile:
                converted_event = self.convert_line(line.strip())
                if converted_event:
                    trace_events.append(converted_event)

        trace_json = {"traceEvents": trace_events}
        with open(self.output_file, "w", encoding='utf-8') as outfile:
            json.dump(trace_json, outfile, ensure_ascii=False, indent=4)
