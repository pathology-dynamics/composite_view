import pandas as pd
import time

from generate_graph import Generate_Graph

stacked_bar_initialization_df = pd.DataFrame(columns=[[
    'initialize_data', 'initialize_simulation_iterations', 'initialize_graph_state', 
    'color_mapping', 'initialize_nx_graph', 'generate_starting_elements', 'generate_table']])

stacked_bar_update_df = pd.DataFrame(columns=[[
    'filter_data', 'trim_graph', 'generate_table', 
    'color_mapping', 'generate_graph_elements']])

for timing_test_iterations in range(1, 11):

    source_id_list = []
    source_name_list = []
    source_type_list = []
    target_id_list = []
    target_name_list = []
    target_type_list = []
    edge_value_list = []

    for i in range(timing_test_iterations * 1000):
        source_id_list.append('node_' + str(i))
        source_name_list.append('node_' + str(i))
        source_type_list.append('source_node')
        target_id_list.append('target_node')
        target_name_list.append('target_node')
        target_type_list.append('target_node')
        
        if i % 2 == 0:
            edge_value_list.append(0.25)
        else:
            edge_value_list.append(0.75)

    data = {
        'source_id': source_id_list, 
        'source_name': source_name_list, 
        'source_type': source_type_list, 
        'target_id': target_id_list, 
        'target_name': target_name_list, 
        'target_type': target_type_list, 
        'edge_value': edge_value_list
    }

    testing_edges = pd.DataFrame(data)

    global_bar_chart_timing = time.time()

    graph = Generate_Graph(edges_df=testing_edges)

    timing_list_initialization = graph.graph_initialization_method_timing_list

    temp_series = pd.Series(timing_list_initialization, index=stacked_bar_initialization_df.columns)
    stacked_bar_initialization_df = stacked_bar_initialization_df.append(temp_series, ignore_index=True)

    #######################################################################
    # Update

    graph.combined_value_range = [0.5, 1]
    graph.graph_update_shortcut = False
    graph.update_graph_elements()

    timing_list_update = graph.graph_update_method_timing_list

    temp_series = pd.Series(timing_list_update, index=stacked_bar_update_df.columns)
    stacked_bar_update_df = stacked_bar_update_df.append(temp_series, ignore_index=True)

    #######################################################################

stacked_bar_initialization_df.to_csv('initialization_runtime_bar.csv')
stacked_bar_update_df.to_csv('update_runtime_bar.csv')