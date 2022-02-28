import numpy as np
import pandas as pd

source_id_list = []
source_name_list = []
source_type_list = []
target_id_list = []
target_name_list = []
target_type_list = []
edge_value_list = []

target_node_count = 1

for i in range(7):
    for j in range(target_node_count):
        source_id_list.append('source_node_' + str(i))
        source_name_list.append('S' + str(i))
        source_type_list.append('source_node')
        target_id_list.append('target_node')
        target_name_list.append('T' + str(j))
        target_type_list.append('target_node')
        edge_value_list.append(1)

data = {
    'source_id': source_id_list, 
    'source_name': source_name_list, 
    'source_type': source_type_list, 
    'target_id': target_id_list, 
    'target_name': target_name_list, 
    'target_type': target_type_list, 
    'edge_value': edge_value_list
}

dummy_df = pd.DataFrame(data)

dummy_df.to_csv('simple_visualization.csv')