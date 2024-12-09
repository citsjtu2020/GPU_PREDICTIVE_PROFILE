import pandas as pd
import torch
import torch.fx
from typing import Any, Callable, Dict, Optional, Tuple
from compute_graph_analysis import utils
from torch.fx import Graph
from torch.fx import map_arg


# from sympy import public

class QualifiedTracer(torch.fx.Tracer):
    """
    ModulePathTracer is an FX tracer that--for each operation--also records
    the qualified name of the Module from which the operation originated.
    """

    # The current qualified name of the Module being traced. The top-level
    # module is signified by empty string. This is updated when entering
    # call_module and restored when exiting call_module
    current_module_qualified_name : str = ''
    # A map from FX Node to the qualname of the Module from which it
    # originated. This is recorded by `create_proxy` when recording an
    # operation
    node_to_originating_module : Dict[torch.fx.Node, str] = {}

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any],
                    args : Tuple[Any, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Override of Tracer.call_module (see
        https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module).

        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Installs the qualified name of the caller in `current_module_qualified_name`
           for retrieval by `create_proxy`
        3) Delegates into the normal Tracer.call_module method
        4) Restores the caller's qualified name into current_module_qualified_name
        """
        old_qualname = self.current_module_qualified_name
        try:
            self.current_module_qualified_name = self.path_of_module(m)
            return super().call_module(m, forward, args, kwargs)
        finally:
            self.current_module_qualified_name = old_qualname

    def create_proxy(self, kind: str, target: torch.fx.node.Target, args: Tuple[Any, ...],
                     kwargs: Dict[str, Any], name: Optional[str] = None, type_expr: Optional[Any] = None):
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_originating_module`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_originating_module[proxy.node] = self.current_module_qualified_name
        return proxy

    def catch_module_info(self,graph: Graph):
        out_res = {"name": [], "op_type": [], "module_qualname": []}
        for node in graph.nodes:
            module_qualname = self.node_to_originating_module.get(node)
            out_res["name"].append(node.name)
            out_res["op_type"].append(node.op)
            out_res["module_qualname"].append(module_qualname)

        out_res_pdf = pd.DataFrame(out_res)
        return out_res_pdf

    # def in_depth_for_sub_module(self,graph:Graph):
    #     for node in graph.nodes:
    #         # Find `call_module` Node in `m` that corresponds to `self.relu`.
    #         # This is the Node we want to swap out for an inlined version of the
    #         # same call
    #         tracer = torch.fx.proxy.GraphAppendingTracer(graph)
    #         if (node.op == "call_module" and node.name in self.node_to_originating_module):
    #             if self.node_to_originating_module[node.name]:
    #                 with graph.inserting_before(node):
    #                     # Create a Proxy from each Node in the current Node's
    #                     # args/kwargs
    #                     proxy_args = map_arg(node.args, lambda n: torch.fx.Proxy(n, tracer))
    #                     proxy_kwargs = map_arg(node.kwargs, lambda n: torch.fx.Proxy(n, tracer))
    #
    #
    #                 proxy_ouput = relu(*proxy_args, **proxy_kwargs)
    #                 # Replace the relu `call_module` node with the inlined
    #                 # version of the function
    #                 node.replace_all_uses_with(proxy_ouput.node)
    #                 # Make sure that the old relu Node is erased
    #                 m.graph.erase_node(node)

    def static_graph_analysis(self,graph: Graph,out_qualified=True,traver_m_dict={}):
        basic_prop_pdf = utils.catch_graph_basic_prop(graph)
        if out_qualified or traver_m_dict:
            qualified_res_pdf = self.catch_module_info(graph=graph)

            basic_prop_pdf = pd.merge(basic_prop_pdf,qualified_res_pdf,on=["op_type","name"],how="left")

        if traver_m_dict:
            detail_info = {"module_qualname":[],"detailed_op":[]}
            for k,v in traver_m_dict.items():
                detail_info["module_qualname"].append(k)
                detail_info["detailed_op"].append(v)

            detail_info_pdf = pd.DataFrame(detail_info)
            detail_info_pdf["op_type"] = "call_module"

            basic_prop_pdf = pd.merge(basic_prop_pdf,detail_info_pdf,on=["op_type","module_qualname"],how="left")

        return basic_prop_pdf





