import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
import time
import json

from colour import Color

class Generate_Graph:
    def __init__(self, edges_df=None):
        
        self._initialize_data(edges_df)

        self._initialize_graph_state()

        self._generate_color_mapping()

        self.nx_graph_initial, self.nx_spring_initial = self._generate_nx_graphs()

        self.starting_elements = self._generate_graph_elements()

        self.table_data_initial = self._generate_table()

    def _initialize_data(self, starting_edges_df=pd.DataFrame):
        """
        Assigns either dummy data or custom data to main dataframe.

        Args:
            starting_edges_df (pandas dataframe): Formatted data input.

        """

        if isinstance(starting_edges_df, pd.DataFrame):
            pass
        
        else:
            starting_edges_df = self._generate_dummy_data()

        self.edges_df = starting_edges_df.sort_values(by='edge_value', ascending=False)
        self.edges_df_initial = starting_edges_df

    def _generate_dummy_data(self):
        """
        Generates formatted dummy data, in the case that other data isn't initially used.

        """

        start_time = time.time()

        base_names = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        relationship_list = [{'T1', 'T2'}, {'T2', 'T3'}, {'T2', 'T3', 'T4'}, {'T4', 'T5'}, {'T5', 'T6'}, {'T6', 'T7'}, {'T7', 'T8'}, {'T7', 'T4'}]
        type_list = ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'type_6', 'type_7', 'type_8']

        source_id_list = []
        source_name_list = []
        source_type_list = []
        target_id_list = []
        target_name_list = []
        target_type_list = []
        edge_value_list = []

        for i in range(len(relationship_list)):
            for name in base_names:
                source_id = str(name) + '_' + str(i)
                source_name = str(name) + '_name'
                source_type = type_list[np.random.randint(0, len(type_list))]

                for target in relationship_list[i]:
                    temp_random_val = np.abs(np.random.normal(0, 0.5))

                    if temp_random_val > 1:
                        temp_random_val = 1
                    if temp_random_val < 0:
                        temp_random_val = 0

                    target_id = str(target)
                    target_name = str(target) + '_name'
                    target_type = 'target_node'
                    edge_val = temp_random_val

                    source_id_list.append(source_id)
                    source_name_list.append(source_name)
                    source_type_list.append(source_type)
                    target_id_list.append(target_id)
                    target_name_list.append(target_name)
                    target_type_list.append(target_type)
                    edge_value_list.append(edge_val)

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

        print('GENERATE DUMMY DATA: ' + str(time.time() - start_time))
        
        return dummy_df

    def _initialize_graph_state(self):
        """
        Initializes attributes/dictionaries that will be used throughout visualizer run.

        """

        start_time = time.time()

        self.max_edge_size = 1
        self.min_edge_size = 0.01

        self.max_node_size = 25
        self.max_node_size_initial = self.max_node_size
        self.min_node_size = 5

        self.node_size_modifier = 0.1
        self.node_size_modifier_initial = self.node_size_modifier

        self.edge_size_modifier = 0.1
        self.edge_size_modifier_initial = self.edge_size_modifier

        self.simulation_iterations = 10
        self.simulation_iterations_initial = self.simulation_iterations

        self.max_edge_value = self.edges_df_initial['edge_value'].max()
        self.min_edge_value = self.edges_df_initial['edge_value'].min()

        self.max_combined_value = 0
        self.min_combined_value = np.inf

        self.source_id_combined_scores_dict = {}

        for node in list(self.edges_df_initial['source_id'].unique()):
            sub_value_list = self.edges_df_initial[self.edges_df_initial['source_id'] == node]['edge_value'].to_list()
            combined_value = self._combine_values(sub_value_list)

            self.source_id_combined_scores_dict[node] = combined_value

            if combined_value > self.max_combined_value:
                self.max_combined_value = combined_value
            
            if combined_value < self.min_combined_value:
                self.min_combined_value = combined_value

        self.source_id_combined_scores_dict_initial = self.source_id_combined_scores_dict

        self.target_size = self.max_combined_value + 0.5 * (self.max_combined_value)
        self.target_size_initial = self.target_size

        self.unique_source_nodes = list(self.edges_df_initial['source_id'].unique())
        self.unique_source_nodes_initial = self.unique_source_nodes

        self.unique_target_nodes = list(self.edges_df_initial['target_id'].unique())
        self.unique_target_nodes_initial = self.unique_target_nodes

        self.source_types = list(self.edges_df_initial['source_type'].unique())
        self.source_types_initial = self.source_types

        self.target_types = list(self.edges_df_initial['target_type'].unique())
        self.target_types_initial = self.target_types

        self.all_types = self.source_types_initial + self.target_types_initial

        self.source_id_name_dict = dict(zip(self.edges_df_initial['source_id'], self.edges_df_initial['source_name']))
        self.source_id_name_dict_initial = self.source_id_name_dict

        self.source_name_id_dict = {}
        for i, j in self.source_id_name_dict_initial.items():
            self.source_name_id_dict.setdefault(j, set()).add(i)

        self.source_name_id_dict_initial = self.source_name_id_dict

        self.target_id_name_dict = dict(zip(self.edges_df_initial['target_id'], self.edges_df_initial['target_name']))
        self.target_id_name_dict_initial = self.target_id_name_dict

        self.target_name_id_dict = {}
        for i, j in self.target_id_name_dict_initial.items():
            self.target_name_id_dict.setdefault(j, set()).add(i)

        self.target_name_id_dict_initial = self.target_name_id_dict

        self.unique_id_name_dict = {**self.source_id_name_dict_initial, **self.target_id_name_dict_initial}
        self.unique_id_name_dict_initial = self.unique_id_name_dict

        self.unique_id_data_dict = {}
        source_sub_df = self.edges_df_initial[['source_id', 'source_name', 'source_type']].drop_duplicates()
        for _, row in source_sub_df.iterrows():
            self.unique_id_data_dict[row['source_id']] = {
                'name': row['source_name'], 
                'type': row['source_type'], 
                'sn_or_tn': 'source_node'
            }

        target_sub_df = self.edges_df_initial[['target_id', 'target_name', 'target_type']].drop_duplicates()
        for _, row in target_sub_df.iterrows():
            self.unique_id_data_dict[row['target_id']] = {
                'name': row['target_name'], 
                'type': row['target_type'], 
                'sn_or_tn': 'target_node'
            }

        print('Intermediate initialize graph state: ' + str(time.time() - start_time))
        '''
        self.source_target_edge_value_dict = {}
        for node in list(self.edges_df_initial['source_id'].unique()):
            if node not in self.source_target_edge_value_dict:
                self.source_target_edge_value_dict[node] = {}

                for target_val_tuple in list(self.edges_df_initial[self.edges_df_initial['source_id'] == node][['target_id', 'edge_value']].to_records(index=False)):
                    self.source_target_edge_value_dict[node][target_val_tuple[0]] = target_val_tuple[1]

        self.source_target_edge_value_dict_initial = self.source_target_edge_value_dict
        '''
        self.source_target_edge_value_dict = {}
        for _, row in self.edges_df_initial.iterrows():
            if row['source_id'] not in self.source_target_edge_value_dict:
                self.source_target_edge_value_dict[row['source_id']] = {row['target_id']: row['edge_value']}

            else:
                self.source_target_edge_value_dict[row['source_id']][row['target_id']] = row['edge_value']

        self.source_target_edge_value_dict_initial = self.source_target_edge_value_dict

        self.unique_target_nodes = list(self.edges_df_initial['target_id'].unique())

        print('Intermediate initialize graph state 2: ' + str(time.time() - start_time))

        # Combined value range initialization
        self.combined_value_range = [self.min_edge_value, self.max_edge_value]
        self.combined_value_range_initial = self.combined_value_range
        self.combined_value_step_size = np.round((self.max_combined_value - self.min_combined_value) / 100, 3)
        self.combined_value_bound = [(self.min_edge_value - self.combined_value_step_size), (self.max_edge_value + self.combined_value_step_size)]

        # Edge value range initialization
        self.edge_value_range = [self.min_edge_value, self.max_edge_value]
        self.edge_value_range_initial = self.edge_value_range
        self.edge_value_step_size = np.round((self.max_edge_value - self.min_edge_value) / 100, 3)
        self.edge_value_bound = [(self.min_edge_value - self.edge_value_step_size), (self.max_edge_value + self.edge_value_step_size)]

        # Max node initialization
        self.max_node_count = len(self.edges_df_initial['source_id'].unique())
        self.max_node_count_initial = self.max_node_count

        # Target filtering initialization
        self.target_dropdown_options = []
        for name in self.target_name_id_dict:
            self.target_dropdown_options.append({'label': name, 'value': name})

        self.target_dropdown_options_initial = self.target_dropdown_options

        self.target_dropdown_selection = []
        self.target_dropdown_selection_initial = self.target_dropdown_selection     

        # Source filtering initialization
        self.source_dropdown_options = []
        for name in self.source_name_id_dict:
            self.source_dropdown_options.append({'label': name, 'value': name})

        self.source_dropdown_options_initial = self.source_dropdown_options

        self.source_dropdown_selection = []
        self.source_dropdown_selection_initial = self.source_dropdown_selection   

        # Type filtering initialization
        self.type_dropdown_options = []
        for type in self.all_types:
            self.type_dropdown_options.append({'label': type, 'value': type})
        
        self.type_dropdown_options_initial = self.type_dropdown_options

        self.type_dropdown_selection = []
        self.type_dropdown_selection_initial = self.type_dropdown_selection

        # TN/SN spread initialization
        self.target_spread = 1
        self.target_spread_initial = self.target_spread
        self.target_spread_previous = 1

        self.source_spread = 0.1
        self.source_spread_initial = self.source_spread
        self.source_spread_previous = 0.1

        # Gradient start initialization
        self.gradient_start = '#272B30'
        self.gradient_color_primacy = False
        self.gradient_start_initial = self.gradient_start

        # Gradient end initialization
        self.gradient_end = '#272B30'
        self.gradient_end_initial = self.gradient_end    

        # SN type color initialization
        self.selected_types = set()
        self.selected_type_color = '#272B30'
        self.type_color_primacy = False
        self.selected_type_color_initial = self.selected_type_color

        # Source color initialization
        self.source_color = '#272B30'
        self.source_color_primacy = False
        self.source_color_initial = self.source_color

        # Target color initialization
        self.target_color = '#272B30'
        self.target_color_primacy = False
        self.target_color_initial = self.target_color

        # Randomized color initialization
        self.random_color_primacy = True

        # Shortcut attribute
        self.graph_update_shortcut = True

        print('INITIALIZE GRAPH STATE: ' + str(time.time() - start_time))
        
    def _initialize_dynamic_attributes(self):
        """
        Initializes attributes (and initial states) that will be used throughout visualizer run.

        """

        start_time = time.time()

        self.max_node_count = None
        self.combined_value_range = [None, None]
        self.edge_value_range = [None, None]

        self.source_dropdown_options = None
        self.target_dropdown_options = None
        self.type_dropdown_options = None

        print('INITIALIZE DYNAMIC ATTRIBUTES: ' + str(time.time() - start_time))

    def _filter_data(self):

        start_time = time.time()

        temporary_full_edges_df = self.edges_df_initial.copy()

        if self.edge_value_range != self.edge_value_range_initial:
            temporary_full_edges_df = temporary_full_edges_df[(temporary_full_edges_df['edge_value'] >= self.edge_value_range[0]) & (temporary_full_edges_df['edge_value'] <= self.edge_value_range[1])]
        
        if self.target_dropdown_selection != self.target_dropdown_selection_initial:
            temporary_full_edges_df = temporary_full_edges_df[temporary_full_edges_df['target_name'].isin(self.target_dropdown_selection)]
        
        if self.source_dropdown_selection != self.source_dropdown_selection_initial:
            temporary_full_edges_df = temporary_full_edges_df[temporary_full_edges_df['source_name'].isin(self.source_dropdown_selection)]

        if self.type_dropdown_selection != self.type_dropdown_selection_initial:
            temporary_full_edges_df = temporary_full_edges_df[temporary_full_edges_df['source_type'].isin(self.type_dropdown_selection)]
            temporary_full_edges_df = temporary_full_edges_df[temporary_full_edges_df['target_type'].isin(self.type_dropdown_selection)]

        self.source_id_combined_scores_dict = {}
        for node in list(temporary_full_edges_df['source_id'].unique()):
            sub_value_list = temporary_full_edges_df[temporary_full_edges_df['source_id'] == node]['edge_value'].to_list()
            combined_value = self._combine_values(sub_value_list)

            if (combined_value <= self.combined_value_range[1]) and (combined_value >= self.combined_value_range[0]):
                self.source_id_combined_scores_dict[node] = combined_value

        self.source_id_combined_scores_df = pd.DataFrame(self.source_id_combined_scores_dict.items(), columns=['source_id', 'combined_value']).sort_values(by=['combined_value'], ascending=False)

        temporary_full_edges_df = temporary_full_edges_df[temporary_full_edges_df['source_id'].isin(self.source_id_combined_scores_df.head(self.max_node_count)['source_id'].to_list())]
        '''
        self.source_target_edge_value_dict = {}
        for node in list(temporary_full_edges_df['source_id'].unique()):
            if node not in self.source_target_edge_value_dict:
                self.source_target_edge_value_dict[node] = {}

                for target_val_tuple in list(temporary_full_edges_df[temporary_full_edges_df['source_id'] == node][['target_id', 'edge_value']].to_records(index=False)):
                    self.source_target_edge_value_dict[node][target_val_tuple[0]] = target_val_tuple[1]
        '''
        self.unique_target_nodes = list(temporary_full_edges_df['target_id'].unique())

        self.edges_df = temporary_full_edges_df

        print('FILTER DATA: ' + str(time.time() - start_time))

    def _trim_graph(self):

        start_time = time.time()

        if self.target_spread != self.target_spread_previous:
            self._generate_nx_graphs()
            self.target_spread_previous = self.target_spread
            # print('1')

        elif self.source_spread != self.source_spread_previous:
            self._generate_nx_graphs()
            self.source_spread_previous = self.source_spread
            # print('2')
        
        elif self.edges_df.shape[0] > len(self.adjusted_nx_graph.edges):
            self._generate_nx_graphs()
            # print('3')
            
        else:
            if not self.graph_update_shortcut:
                remaining_edges_df = self.edges_df[['source_id', 'target_id']]
                total_edges_df = self.edges_df_initial[['source_id', 'target_id']]

                remove_edges_df = total_edges_df[~total_edges_df.apply(tuple, 1).isin(remaining_edges_df.apply(tuple, 1))]

                remove_edges_list = remove_edges_df.to_records(index=False)

                remove_edges_list_formatted = []

                for edge in remove_edges_list:
                    remove_edges_list_formatted.append(tuple(edge))

                temporary_graph = self.adjusted_nx_graph.copy()
                temporary_graph.remove_edges_from(remove_edges_list_formatted)
                temporary_graph.remove_nodes_from(list(nx.isolates(temporary_graph)))
                
                self.nx_graph = temporary_graph

                # print('4')
            
            # print('5')

        print('TRIM GRAPH: ' + str(time.time() - start_time))

    def _combine_values(self, values=list):

        number_of_connections = len(values)

        linear_combination = 0
        for value in values:
            linear_combination = linear_combination + (value / number_of_connections)

        return linear_combination

    def _generate_size(self, value, node=True):

        if node:
            return (self.min_node_size + ((self.max_node_size - self.min_node_size) * ((value - self.min_combined_value) / (self.max_combined_value - self.min_combined_value)))) * self.node_size_modifier

        else:
            return self.min_edge_size + ((self.max_edge_size - self.min_edge_size) * ((value - self.min_edge_value) / (self.max_edge_value - self.min_edge_value))) * self.edge_size_modifier * 0.3

    def _generate_color_mapping(self):

        start_time = time.time()

        if self.random_color_primacy:
            self.type_color_dict = {}
            color_intervals = (330 / len(self.all_types))
            random_color_list = []

            for i in range(len(self.all_types)):

                normal_val = np.abs(np.random.normal(0.5, 0.33, 1)[0])
                if normal_val > 1 : normal_val = 1

                normal_color = color_intervals * normal_val
                random_color_list.append('hsl(' + str((color_intervals * i) + normal_color) + ', 100%, 60%)')

            random_color_list = np.random.choice(random_color_list, len(self.all_types), replace=False)

            for i, type in enumerate(self.all_types):
                self.type_color_dict[type] = random_color_list[i]

            self.random_color_primacy = False
        
        if self.gradient_color_primacy:
            starting_color = Color(self.gradient_start)
            color_gradient_list = list(starting_color.range_to(Color(self.gradient_end), len(self.all_types)))

            for i, type in enumerate(self.all_types):
                self.type_color_dict[type] = str(color_gradient_list[i])

            self.gradient_color_primacy = False

        if self.type_color_primacy:
            for type in self.selected_types:
                self.type_color_dict[type] = self.selected_type_color

            self.type_color_primacy = False

        if self.source_color_primacy:
            for type in self.type_color_dict:
                if type not in self.target_types:
                    self.type_color_dict[type] = self.source_color

            self.source_color_primacy = False

        if self.target_color_primacy:
            for type in self.type_color_dict:
                if type in self.target_types:
                    self.type_color_dict[type] = self.target_color

            self.target_color_primacy = False

        print('GENERATE COLORS: ' + str(time.time() - start_time))

    def _generate_nx_graphs(self):

        start_time = time.time()

        nx_graph = nx.from_pandas_edgelist(self.edges_df, 'source_id', 'target_id')

        combination_set = set(list(combinations(self.unique_target_nodes, 2)))

        connected_targets = {}
        sn_targets_connections = {}

        for i, combination in enumerate(combination_set):
            paths = list(nx.all_simple_paths(nx_graph, combination[0], combination[1], cutoff=3))

            if len(paths) != 0:
                edge_count = 0

                for path in paths:
                    if len(path) == 3:
                        edge_count = edge_count + 1

                        if path[1] in sn_targets_connections:
                            sn_targets_connections[path[1]] = set.union(sn_targets_connections[path[1]], {combination[0], combination[1]})
                        else:
                            sn_targets_connections[path[1]] = {combination[0], combination[1]}

                connected_targets['combination_' + str(i)] = {'combination': combination, 'edge_count': edge_count}

        one_degree_sources = {}
        for target in self.unique_target_nodes:
            for neighbor in nx_graph.neighbors(target):
                if nx_graph.degree(neighbor) == 1:
                    one_degree_sources[neighbor] = target
        
        max_edges = 0
        min_edges = np.inf

        initial_graph = nx.Graph()

        for combination_number in connected_targets:
            edge_count = connected_targets[combination_number]['edge_count']

            if edge_count > max_edges:
                max_edges = edge_count

            if edge_count < min_edges:
                min_edges = edge_count

        for combination_number in connected_targets:
            edge_count = connected_targets[combination_number]['edge_count']
            combination = connected_targets[combination_number]['combination']

            if max_edges != min_edges:
                edge_weight = 7 - 3 * ((edge_count - min_edges) / (max_edges - min_edges))
            else:
                edge_weight = 7

            initial_graph.add_edge(combination[0], combination[1], weight=edge_weight)

        for target in self.unique_target_nodes:
            if target not in initial_graph:
                initial_graph.add_node(target)

        initial_spring = nx.spring_layout(initial_graph, dim=2, weight='weight', k=self.target_spread, iterations=50)

        pos_dict = {}

        max_x_global = -np.inf
        max_y_global = -np.inf

        sn_centroid_dict = {}
        for node in sn_targets_connections:
            connected_targets_list = [str(s) for s in sn_targets_connections[node]]
            connected_targets_string = "-".join(connected_targets_list)

            if connected_targets_string not in sn_centroid_dict:
                middle_x = 0
                middle_y = 0

                min_x = np.inf
                min_y = np.inf

                max_x = -np.inf
                max_y = -np.inf

                for connected_target in sn_targets_connections[node]:
                    middle_x = middle_x + initial_spring[connected_target][0]
                    middle_y = middle_y + initial_spring[connected_target][1]

                    if initial_spring[connected_target][0] > max_x:
                        max_x = initial_spring[connected_target][0]
                    
                    if initial_spring[connected_target][0] < min_x:
                        min_x = initial_spring[connected_target][0]

                    if initial_spring[connected_target][1] > max_y:
                        max_y = initial_spring[connected_target][1]
                    
                    if initial_spring[connected_target][1] < min_y:
                        min_y = initial_spring[connected_target][1]

                middle_x = middle_x / len(sn_targets_connections[node])
                middle_y = middle_y / len(sn_targets_connections[node])

                min_range = min([(max_x - min_x), (max_y - min_y)])

                sn_centroid_dict[connected_targets_string] = (middle_x, middle_y, min_range)

                x_pos = np.random.normal(middle_x, (np.abs(min_range) / 8), 1)[0]
                y_pos = np.random.normal(middle_y, (np.abs(min_range) / 8), 1)[0]

                if max_x_global < max_x:
                    max_x_global = max_x

                if max_y_global < max_y:
                    max_y_global = max_y

                pos_dict[node] = (x_pos, y_pos)

            else:
                middle_x, middle_y, min_range = sn_centroid_dict[connected_targets_string]

                x_pos = np.random.normal(middle_x, (np.abs(min_range) / 8), 1)[0]
                y_pos = np.random.normal(middle_y, (np.abs(min_range) / 8), 1)[0]

                pos_dict[node] = (x_pos, y_pos)

        for one_degree_source in one_degree_sources:
            x_pos = initial_spring[one_degree_sources[one_degree_source]][0]
            y_pos = initial_spring[one_degree_sources[one_degree_source]][1]

            if (max_x_global == -np.inf) and (max_y_global == -np.inf):
                gaussian_range = 1
            
            else:
                gaussian_range = min([max_x_global, max_y_global]) / 4

            x_pos_new = np.random.normal(x_pos, (np.abs(gaussian_range) / 8), 1)[0]
            y_pos_new = np.random.normal(y_pos, (np.abs(gaussian_range) / 8), 1)[0]

            pos_dict[one_degree_source] = (x_pos_new, y_pos_new)

        fixed_list = []

        for entry in initial_graph.nodes:
            pos_dict[entry] = [initial_spring[entry][0], initial_spring[entry][1]]
            fixed_list.append(entry)
        
        if (len(pos_dict) != 0) and (len(fixed_list) != 0):
            final_spring = nx.spring_layout(nx_graph, dim=2, pos=pos_dict, fixed=fixed_list, k=self.source_spread, iterations=self.simulation_iterations)
        else:
            final_spring = nx.spring_layout(nx_graph, dim=2, k=0.05, iterations=10)

        self.adjusted_nx_graph = nx_graph
        self.nx_graph = nx_graph
        self.final_spring = final_spring

        print('GENERATE NX GRAPH: ' + str(time.time() - start_time))

        return (nx_graph, final_spring)

    def _generate_graph_elements(self):

        start_time = time.time()

        elements = []
        
        for node in self.nx_graph.nodes:
            if node in self.unique_target_nodes:
                node_value = 'None'
                size_val = self._generate_size(self.target_size)

            else:
                node_value = self.source_id_combined_scores_dict[node]
                size_val = self._generate_size(node_value)
                
            elements.append({
                'data': {
                    'id': node, 
                    'label': self.unique_id_data_dict[node]['name'], 
                    'value': node_value,
                    'type': self.unique_id_data_dict[node]['type'], 
                    'size': size_val, 
                    'label_size': size_val * 0.3,
                    'color': self.type_color_dict[self.unique_id_data_dict[node]['type']],
                    'sn_or_tn': self.unique_id_data_dict[node]['sn_or_tn']
                }, 
                'position': {'x': 100 * self.final_spring[node][0], 'y': 100 * self.final_spring[node][1]}
            })

            if self.unique_id_data_dict[node]['sn_or_tn'] == 'source_node':
                if self.nx_graph.degree(node) != 0:
                    for target in self.nx_graph.neighbors(node):
                        edge_val = np.round(self.source_target_edge_value_dict[node][target], 3)
                        elements.append({
                            'data': {
                                'source': node, 
                                'target': target, 
                                'label': edge_val,
                                'size': self._generate_size(edge_val, node=False), 
                                'label_size': self._generate_size(edge_val, node=False) * 10
                            }
                        })

        self.elements = elements

        print('GENERATE ELEMENTS: ' + str(time.time() - start_time))

        return elements

    def _generate_table(self):

        start_time = time.time()

        table_df = self.edges_df.copy()

        table_df['combined_source_value'] = table_df['source_id']

        table_df = table_df.replace({"combined_source_value": self.source_id_combined_scores_dict})

        table_df = table_df.round(3)
        
        self.table_data = table_df
        self.data_table_columns = [{"name": i, "id": i} for i in table_df.columns]
        self.table_data = table_df.to_dict('records')

        print('GENERATE TABLE: ' + str(time.time() - start_time))

        return self.table_data

    def generate_node_data(self, selected_nodes_list):

        start_time = time.time()

        formatted_data_list = []
        self.selected_types = set()

        for node in selected_nodes_list:
            if node['sn_or_tn'] == 'source_node':
                self.selected_types.add(node['type'])

                edges = {}
                for _, connecting_node in enumerate(self.nx_graph[node['id']]):
                    edges[str(self.unique_id_data_dict[connecting_node]['name']) + ' (ID:' + str(connecting_node) + ')'] = float(np.round(self.source_target_edge_value_dict[node['id']][connecting_node], 3))

                edges_sorted = dict(sorted(edges.items(), key=lambda item: item[1], reverse=True))

                data_dump = {
                    'source_id': node['id'], 
                    'source_name': node['label'], 
                    'node_type': node['type'], 
                    'combined_value': float(np.round(self.source_id_combined_scores_dict[node['id']], 3)), 
                    'sn_or_tn': 'source_node', 
                    'edges': edges_sorted
                    }

                formatted_data_list.append(json.dumps(data_dump, indent=2))
            
            if node['sn_or_tn'] == 'target_node':

                edges = {}
                for _, connecting_node in enumerate(self.nx_graph[node['id']]):
                    edges[str(self.unique_id_data_dict[connecting_node]['name']) + ' (ID:' + str(connecting_node) + ')'] = float(np.round(self.source_target_edge_value_dict[connecting_node][node['id']], 3))
                
                edges_sorted = dict(sorted(edges.items(), key=lambda item: item[1], reverse=True))

                data_dump = {
                    'node_id': node['id'], 
                    'node_name': node['label'], 
                    'sn_or_tn': 'target_node', 
                    'edges': edges_sorted
                    }

                formatted_data_list.append(json.dumps(data_dump, indent=2))

        print('NODE ELEMENTS: ' + str(time.time() - start_time))
            
        return formatted_data_list

    def update_graph_elements(self):

        if not self.graph_update_shortcut:
            self._filter_data()

        self._trim_graph()

        if not self.graph_update_shortcut:
            self._generate_table()

        self._generate_color_mapping()

        self._generate_graph_elements()

        self.graph_update_shortcut = True

        return self.elements        

    def reset_graph(self):

        start_time = time.time()

        self.combined_value_range = self.combined_value_range_initial
        self.edge_value_range = self.edge_value_range_initial
        self.max_node_count = self.max_node_count_initial

        self.target_dropdown_selection = self.target_dropdown_selection_initial
        self.target_dropdown_options = self.target_dropdown_options_initial

        self.source_dropdown_selection = self.source_dropdown_selection_initial
        self.source_dropdown_options = self.source_dropdown_options_initial

        self.type_dropdown_selection = self.type_dropdown_selection_initial
        self.type_dropdown_options = self.type_dropdown_options_initial

        self.target_spread = self.target_spread_initial
        self.target_spread_previous = self.target_spread_initial

        self.source_spread = self.source_spread_initial
        self.source_spread_previous = self.source_spread_initial

        self.node_size_modifier = self.node_size_modifier_initial
        self.edge_size_modifier = self.edge_size_modifier_initial
        self.simulation_iterations = self.simulation_iterations_initial

        self.gradient_start = self.gradient_start_initial
        self.gradient_end = self.gradient_end_initial
        self.selected_type_color = self.selected_type_color_initial
        self.source_color = self.source_color_initial
        self.target_color = self.target_color_initial

        self.edges_df = self.edges_df_initial

        self.source_id_combined_scores_dict = self.source_id_combined_scores_dict_initial

        self.table_data = self.table_data_initial

        self.nx_graph = self.nx_graph_initial
        self.final_spring = self.nx_spring_initial
        self.adjusted_nx_graph = self.nx_graph_initial

        print('RESET GRAPH: ' + str(time.time() - start_time))

    def load_additional_data(self, df_input):

        start_time = time.time()
        
        self._initialize_data(df_input)

        self._initialize_graph_state()

        self._filter_data()

        self._generate_color_mapping()

        self.nx_graph_initial, self.nx_spring_initial = self._generate_nx_graphs()
        
        self.starting_elements = self._generate_graph_elements()

        self.table_data_initial = self._generate_table()

        print('LOAD ADDTL DATA: ' + str(time.time() - start_time))

        return self.elements

    def simulate(self):

        # start_time = time.time()

        self._generate_nx_graphs()

        self.adjusted_nx_graph = self.nx_graph

        self._generate_graph_elements()

        # print('SIMULATION: ' + str(time.time() - start_time))

'''
app = dash.Dash()

elements = []

for node in generate_graph.nx_graph:
    elements.append({
        'data': {
            'id': node, 
            'label': node, 
            'size': '0.01px'
        }, 
        'position': {'x': generate_graph.final_spring[node][0], 'y': generate_graph.final_spring[node][1]}
    })
    

for node_1 in generate_graph.nx_graph:
    for node_2 in generate_graph.nx_graph[node_1]:
        elements.append({
            'data': {
                'source': node_1,
                'target': node_2
            }
        })

default_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'width': 'data(size)',
            'height': 'data(size)',
            'content': 'data(label)',
            'font-size': 'data(label_size)'
            }
        },
    {
        'selector': 'edge',
        'style': {
            'width': 'data(size)', 
            'font-opacity': 1
            }
        }, 
    {
        'selector': ':selected',
        'style': {
            'border-color': 'black', 
            'border-opacity': '1', 
            'border-width': '0.3px'
            }
        }
    ]

cytoscape_graph = cyto.Cytoscape(
    id='output_graph',
    layout={'name': 'preset'},
    style={'width': '100vw', 'height': '100vh'},
    stylesheet=default_stylesheet, 
    elements=generate_graph.elements,
    boxSelectionEnabled=True
    )

app.layout = html.Div([
    cytoscape_graph
])

if __name__ == '__main__':
    app.run_server(debug=True)
'''