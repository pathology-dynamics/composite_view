import json
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph
import networkx as nx

from generate_graph import Generate_Graph

graph = Generate_Graph()

graph.combined_value_range = [0.5, 1]
graph.graph_update_shortcut = False
elements = graph.update_graph_elements()

for key in graph.__dict__.keys():
    print(key)
    print(type(graph.__dict__[key]))

'''
json_input = json.load(open('json_data.json'))

graph = Generate_Graph(json_data=json_input)

for key in graph.__dict__.keys():
    print(key)
    print(type(graph.__dict__[key]))

print(len(graph.elements))

graph.combined_value_range = [0.5, 1]
graph.graph_update_shortcut = False
elements = graph.update_graph_elements()

print(len(elements))
'''