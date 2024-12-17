# graph_utils.py

import torch.fx as fx
import pandas as pd
import pygraphviz as pgv

def extract_graph(model):
    tracer = fx.Tracer()
    graph = tracer.trace(model)

    # 创建 GraphModule
    graph_module = fx.GraphModule(model, graph)

    data = []
    adj = {}
    for node in graph.nodes:
        op_type = node.op
        name = node.name
        num_downstream = len(node.users)
        num_upstream = len(node.all_input_nodes)
        downstream = [user.name for user in node.users]
        upstream = [input_node.name for input_node in node.all_input_nodes]
        target = node.target
        args = str(node.args)
        module_qualname = ""
        detailed_op = ""

        # 构建邻接表
        if name not in adj:
            adj[name] = []
        for i in range(len(downstream)):
            adj[name].append(downstream[i])

        if op_type == "call_module":
            module = dict(graph_module.named_modules())[target]
            module_qualname = target
            detailed_op = str(module)

        data.append([
            op_type, name, num_downstream, num_upstream, downstream, upstream,
            target, args, module_qualname, detailed_op
        ])

    static_graph = pd.DataFrame(data, columns=[
        "op_type", "name", "num_downstream", "num_upstream", "downstream", "upstream",
        "target", "args", "module_qualname", "detailed_op"
    ])

    op_type = {}
    name_module = {}
    for i in range(len(static_graph)):
        if static_graph.loc[i, 'detailed_op'] != "":
            key = static_graph.iloc[i]['detailed_op'].split("(")[0]
            if key not in op_type:
                op_type[key] = 0
            module_name = 'nn.Module: ' + key + '_' + str(op_type[key])
            op_type[key] += 1
            static_graph.loc[i, 'module'] = module_name
            name_module[static_graph.iloc[i]['name']] = [static_graph.iloc[i]['detailed_op'], module_name]
        else:
            static_graph.loc[i, 'module'] = static_graph.iloc[i]['name']
            name_module[static_graph.iloc[i]['name']] = [static_graph.iloc[i]['name'], static_graph.iloc[i]['name']]

    # 用 module 名称替换 name
    # adj_module = {}
    # for key in adj:
    #     adj_module[name_module[key][1]] = []
    #     for neighbor in adj[key]:
    #         adj_module[name_module[key][1]].append(name_module[neighbor][1])

    return static_graph, name_module, adj

#可视化计算图
def draw_graph(adj, name_module, start_node='x', model_name='model',t=0):
    def draw(node, G, visited):
        if node not in visited:
            visited.add(node)
            G.add_node(node, label=name_module[node][t])
            if node in adj:
                for neighbor in adj[node]:
                    G.add_node(neighbor, label=name_module[neighbor][t])
                    G.add_edge(node, neighbor, arrowsize=0.6)
                    draw(neighbor, G, visited)

    G = pgv.AGraph(strict=True, directed=True)
    visited = set()
    draw(start_node, G, visited)
    G.graph_attr['splines'] = 'true'
    G.graph_attr['rankdir'] = 'TB'  # 从上到下
    G.node_attr['shape'] = 'box'
    G.node_attr['style'] = 'filled'
    G.node_attr['fillcolor'] = 'lightblue'
    G.node_attr['fontname'] = 'Consolas'
    G.node_attr['fontsize'] = 15
    G.layout(prog='dot')
    #创建一个文件夹，名称为model_name
    output_file = './graph/'+ model_name +'_' +str(t) + '.png'
    G.draw(output_file)

# draw_graph(adj, name_module, output_file='graph_1.png',t=1)
# draw_graph(adj, name_module, output_file='graph_0.png',t=0)