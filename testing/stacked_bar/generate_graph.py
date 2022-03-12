import pandas as pd
import numpy as np
import networkx as nx
from colour import Color
from scipy import stats

from itertools import combinations
import time
import json

class Generate_Graph:
    '''
    Class for creating an interactive graph object for use in visualizer.py.

    '''

    def __init__(self, edges_df=None, timing=False, json_data=False):

        self.timing = timing

        self.graph_initialization_method_timing_list = []

        if not json_data:
            start_time = time.time()
            self._initialize_data(edges_df)
            self.graph_initialization_method_timing_list.append(time.time() - start_time)

            start_time = time.time()
            self._initialize_simulation_iterations()
            self.graph_initialization_method_timing_list.append(time.time() - start_time)

            start_time = time.time()
            self._initialize_graph_state()
            self.graph_initialization_method_timing_list.append(time.time() - start_time)

            start_time = time.time()
            self.type_color_dict_initial = self._generate_color_mapping()
            self.graph_initialization_method_timing_list.append(time.time() - start_time)

            start_time = time.time()
            self.nx_graph_initial, self.nx_spring_initial = self._generate_nx_graphs()
            self.graph_initialization_method_timing_list.append(time.time() - start_time)

            start_time = time.time()
            self.starting_elements = self._generate_graph_elements()
            self.graph_initialization_method_timing_list.append(time.time() - start_time)

            start_time = time.time()
            self.table_data_initial = self._generate_table()
            self.graph_initialization_method_timing_list.append(time.time() - start_time)

        else:
            self._load_attributes(json_data)

    def _load_attributes(self, json_data):
        """
        Allows a graph object to be created with previously generated attributes (in the form of a json dictionary).

        Args:
            json_data (dict): Attributes of the Generate_Graph() class that have been previously determined.

        """

        start_time = time.time()
        
        self.__dict__ = dict(json_data)

        if self.timing:
            print('LOADING ATTRIBUTES: ' + str(time.time() - start_time))

    def _initialize_data(self, starting_edges_df=pd.DataFrame):
        """
        Assigns either dummy data or custom data to main dataframe.

        Args:
            starting_edges_df (Pandas dataframe): Formatted data input.

        """

        start_time = time.time()

        if isinstance(starting_edges_df, pd.DataFrame):
            pass
        
        else:
            starting_edges_df = self._generate_dummy_data()

        self.edges_json = starting_edges_df.sort_values(by='edge_value', ascending=False).to_json()
        self.edges_json_initial = starting_edges_df.to_json()

        if self.timing:
            print('INITIALIZE DATA: ' + str(time.time() - start_time))

    def _initialize_simulation_iterations(self):
        """
        Sets value for number of networkx spring graph simulations. This gets its own helper method 
        for scope reasons.

        """        

        # Value assignment... no timing necessary.
        self.simulation_iterations = 10
        self.simulation_iterations_initial = self.simulation_iterations


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

        if self.timing:
            print('GENERATE DUMMY DATA: ' + str(time.time() - start_time))
        
        return dummy_df

    def _initialize_graph_state(self):
        """
        Initializes attributes/dictionaries that will be used throughout visualizer.

        """

        start_time = time.time()

        edges_df_initial = pd.read_json(self.edges_json_initial)

        self.target_types = list(edges_df_initial['target_type'].unique())
        self.all_types = list(edges_df_initial['source_type'].unique()) + self.target_types

        self.target_id_name_dict = dict(zip(edges_df_initial['target_id'], edges_df_initial['target_name']))
        self.target_id_name_dict_initial = self.target_id_name_dict

        self.max_edge_value = edges_df_initial['edge_value'].max()
        self.min_edge_value = edges_df_initial['edge_value'].min()

        self.max_combined_value = 0
        self.min_combined_value = np.inf

        # Changing this value may help when dealing with very small decimal values.
        self.round_precision = 3

        self.source_id_combined_scores_dict = {}
        for node in list(edges_df_initial['source_id'].unique()):

            sub_df = edges_df_initial[edges_df_initial['source_id'] == node]
            combined_value = self._combine_values(sub_df)

            self.source_id_combined_scores_dict[node] = combined_value

            if combined_value > self.max_combined_value:
                self.max_combined_value = combined_value
            
            if combined_value < self.min_combined_value:
                self.min_combined_value = combined_value

        self.max_combined_value = float(self.max_combined_value)
        self.min_combined_value = float(self.min_combined_value)

        self.max_combined_value_start = self.max_combined_value
        self.min_combined_value_start = self.min_combined_value

        self.source_id_combined_scores_dict_initial = self.source_id_combined_scores_dict

        self.unique_id_data_dict = {}
        source_sub_df = edges_df_initial[['source_id', 'source_name', 'source_type']].drop_duplicates()
        for _, row in source_sub_df.iterrows():
            self.unique_id_data_dict[row['source_id']] = {
                'name': row['source_name'], 
                'type': row['source_type'], 
                'sn_or_tn': 'source_node'
            }

        target_sub_df = edges_df_initial[['target_id', 'target_name', 'target_type']].drop_duplicates()
        for _, row in target_sub_df.iterrows():
            self.unique_id_data_dict[row['target_id']] = {
                'name': row['target_name'], 
                'type': row['target_type'], 
                'sn_or_tn': 'target_node'
            }

        self.source_target_edge_value_dict = {}
        for _, row in edges_df_initial.iterrows():
            if row['source_id'] not in self.source_target_edge_value_dict:
                self.source_target_edge_value_dict[row['source_id']] = {row['target_id']: row['edge_value']}

            else:
                self.source_target_edge_value_dict[row['source_id']][row['target_id']] = row['edge_value']

        self.source_target_edge_value_dict_initial = self.source_target_edge_value_dict

        self.unique_target_nodes = list(edges_df_initial['target_id'].unique())
        self.unique_target_nodes_initial = self.unique_target_nodes

        # Combined value range initialization
        self.combined_value_range = [self.min_combined_value_start, self.max_combined_value_start]
        self.combined_value_range_initial = self.combined_value_range
        self.combined_value_step_size = np.round((self.max_combined_value_start - self.min_combined_value_start) / 100, self.round_precision).item()
        self.combined_value_bound = [(self.min_combined_value_start - self.combined_value_step_size), (self.max_combined_value_start + self.combined_value_step_size)]
        self.combined_value_bound_initial = self.combined_value_bound

        # Edge value range initialization
        self.edge_value_range = [self.min_edge_value, self.max_edge_value]
        self.edge_value_range_initial = self.edge_value_range
        self.edge_value_step_size = np.round((self.max_edge_value - self.min_edge_value) / 100, self.round_precision).item()
        self.edge_value_bound = [(self.min_edge_value - self.edge_value_step_size), (self.max_edge_value + self.edge_value_step_size)]

        # Max node initialization
        self.max_node_count = len(edges_df_initial['source_id'].unique())
        self.max_node_count_initial = self.max_node_count

        # Target filtering initialization
        self.target_dropdown_options = []
        for name in list(edges_df_initial['target_name'].unique()):
            self.target_dropdown_options.append({'label': name, 'value': name})

        self.target_dropdown_options_initial = self.target_dropdown_options

        self.target_dropdown_selection = []
        self.target_dropdown_selection_initial = self.target_dropdown_selection     

        # Source filtering initialization
        self.source_dropdown_options = []
        for name in list(edges_df_initial['source_name'].unique()):
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

        # Node/Edge size initialization
        self.max_edge_size = 1
        self.min_edge_size = 0.01

        self.max_node_size = 25
        self.max_node_size_initial = self.max_node_size
        self.min_node_size = 5

        self.node_size_modifier = 0.1
        self.node_size_modifier_initial = self.node_size_modifier

        self.edge_size_modifier = 0.1
        self.edge_size_modifier_initial = self.edge_size_modifier

        self.target_size = self.max_combined_value_start + 0.5 * (self.max_combined_value_start)
        self.target_size_initial = self.target_size

        # Layout initialization
        self.layout = 'adjusted_spring'

        # Gradient start initialization
        self.gradient_start = '#272B30'
        self.gradient_color_primacy = False
        self.gradient_start_initial = self.gradient_start

        # Gradient end initialization
        self.gradient_end = '#272B30'
        self.gradient_end_initial = self.gradient_end    

        # SN type color initialization
        self.selected_types = []
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

        if self.timing:
            print('INITIALIZE GRAPH STATE: ' + str(time.time() - start_time))

    def _combine_values(self, df=pd.DataFrame, method='arithmetic_mean'):
        """
        Helper method that allows for source-target values to be combined. When a source shares an edge with 
        multiple targets, each edge value for that source is fed into this method. Adjusting this method allows
        for different linear combinations (or otherwise) for data-specific graph customization.

        Args:
            df (Pandas dataframe): Sub-dataframe of edges to be manipulated (usually combining edge values).
            method (str): Stock score combination methods; other combination methods can replace 'other.'

        """
        
        if method == 'arithmetic_mean':

            return df['edge_value'].mean()

        elif method == 'geometric_mean':

            return stats.gmean(df['edge_value'])

        elif method == 'other':

            return df['edge_value'].sum()

    def _filter_data(self):
        """
        Filters main edges dataframe based on slider values. Also adjusts certain dictionaries based 
        on filtered data.

        """

        start_time = time.time()

        temporary_full_edges_df = pd.read_json(self.edges_json_initial).copy()

        if self.edge_value_range != self.edge_value_range_initial:
            temporary_full_edges_df = temporary_full_edges_df[(temporary_full_edges_df['edge_value'] >= self.edge_value_range[0]) & (temporary_full_edges_df['edge_value'] <= self.edge_value_range[1])]
        
        if self.target_dropdown_selection != self.target_dropdown_selection_initial:
            temporary_full_edges_df = temporary_full_edges_df[temporary_full_edges_df['target_name'].isin(self.target_dropdown_selection)]
        
        if self.source_dropdown_selection != self.source_dropdown_selection_initial:
            temporary_full_edges_df = temporary_full_edges_df[temporary_full_edges_df['source_name'].isin(self.source_dropdown_selection)]

        if self.type_dropdown_selection != self.type_dropdown_selection_initial:
            temporary_full_edges_df = temporary_full_edges_df[temporary_full_edges_df['source_type'].isin(self.type_dropdown_selection)]
            temporary_full_edges_df = temporary_full_edges_df[temporary_full_edges_df['target_type'].isin(self.type_dropdown_selection)]

        temp_max_combined_val = -np.inf
        temp_min_combined_val = np.inf

        self.source_id_combined_scores_dict = {}
        for node in list(temporary_full_edges_df['source_id'].unique()):

            sub_df = temporary_full_edges_df[temporary_full_edges_df['source_id'] == node]
            combined_value = self._combine_values(sub_df)

            # Dynamically change combined score bound.
            if combined_value > temp_max_combined_val:
                temp_max_combined_val = combined_value

            if combined_value < temp_min_combined_val:
                temp_min_combined_val = combined_value     

            if (combined_value <= self.combined_value_range[1]) and (combined_value >= self.combined_value_range[0]):
                self.source_id_combined_scores_dict[node] = combined_value

        self.combined_value_bound = [temp_min_combined_val - self.combined_value_step_size, temp_max_combined_val + self.combined_value_step_size]

        source_id_combined_scores_df = pd.DataFrame(self.source_id_combined_scores_dict.items(), columns=['source_id', 'combined_value']).sort_values(by=['combined_value'], ascending=False)

        temporary_full_edges_df = temporary_full_edges_df[temporary_full_edges_df['source_id'].isin(source_id_combined_scores_df.head(self.max_node_count)['source_id'].to_list())]

        self.unique_target_nodes = list(temporary_full_edges_df['target_id'].unique())

        self.edges_json = temporary_full_edges_df.to_json()

        if self.timing:
            print('FILTER DATA: ' + str(time.time() - start_time))

    def _trim_graph(self):
        """
        Runs _generate_nx_graphs() based on user input. If possible, the graph is NOT re-simulated to help 
        with performance. If certain conditions are met, the elements list is filtered instead.

        """

        start_time = time.time()

        edges_df = pd.read_json(self.edges_json)
        adjusted_nx_graph = nx.node_link_graph(self.adjusted_nx_graph)

        if self.target_spread != self.target_spread_previous:
            self._generate_nx_graphs()
            self.target_spread_previous = self.target_spread

        elif self.source_spread != self.source_spread_previous:
            self._generate_nx_graphs()
            self.source_spread_previous = self.source_spread
        
        elif edges_df.shape[0] > len(adjusted_nx_graph.edges):
            self._generate_nx_graphs()
            
        else:
            if not self.graph_update_shortcut:
                remaining_edges_df = edges_df[['source_id', 'target_id']]
                total_edges_df = pd.read_json(self.edges_json_initial)[['source_id', 'target_id']]

                remove_edges_df = total_edges_df[~total_edges_df.apply(tuple, 1).isin(remaining_edges_df.apply(tuple, 1))]

                remove_edges_list = remove_edges_df.to_records(index=False)

                remove_edges_list_formatted = []

                for edge in remove_edges_list:
                    remove_edges_list_formatted.append(tuple(edge))

                temporary_graph = adjusted_nx_graph.copy()
                temporary_graph.remove_edges_from(remove_edges_list_formatted)
                temporary_graph.remove_nodes_from(list(nx.isolates(temporary_graph)))

                self.nx_graph = nx.node_link_data(temporary_graph)

        if self.timing:
            print('TRIM GRAPH: ' + str(time.time() - start_time))

    def _generate_size(self, value=float, node=True):
        """
        Sets node/edge size based on the associated value (combined value for nodes, edge value for edges). 
        Adjusting this method allows for node/edge size flexibility. Node/edge modifier is adjustable in the app, 
        acts as a simple coefficient.

        Args:
            value (float): Value that node/edge size will be based on.
            node (bool): Bool that activates node sized instead of edge sizing.

        Note:
            In the current state, size is determined by this formula:
            
                (min_size + max_min_size_difference * ((value - min_size) / max_min_size_difference)) * additional_size_coefficient

        """

        if node:
            if self.min_combined_value_start == self.max_combined_value_start: 
                return self.min_node_size * self.node_size_modifier

            else:
                return (self.min_node_size + ((self.max_node_size - self.min_node_size) * ((value - self.min_combined_value_start) / (self.max_combined_value_start - self.min_combined_value_start)))) * self.node_size_modifier

        else:
            if self.min_edge_value == self.max_edge_value:
                return 10 * self.min_edge_size * self.edge_size_modifier
            
            else:
                return (self.min_edge_size + ((self.max_edge_size - self.min_edge_size) * ((value - self.min_edge_value) / (self.max_edge_value - self.min_edge_value)))) * self.edge_size_modifier * 0.5

    def _generate_color_mapping(self):
        """
        Creates a dictionary that holds all type:color mapping information. There are additional switches that allow for 
        the dictionary to be edited based on the user's preferences in the app.

        """

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

        if self.timing:
            print('GENERATE COLORS: ' + str(time.time() - start_time))

        return self.type_color_dict

    def _generate_nx_graphs(self):
        """
        Creates a networkx graph based on the filtered main edges dataframe. If applicable, simulations are performed 
        (using the Fruchterman-Reingold force-directed algorithm) to improve node layout and spacing. Additional layout 
        options are provided, however change self.layout at your own risk! All of the sliders don't necessarily work with 
        each of the different layouts. Additionally, the adjusted_spring layout is tailor-made for this visualizer/data structure.

        """

        start_time = time.time()

        temp_df = pd.read_json(self.edges_json)[['source_id', 'target_id', 'edge_value']].copy()

        if self.layout == 'random':
            nx_graph = nx.from_pandas_edgelist(temp_df, 'source_id', 'target_id')

            final_spring = nx.random_layout(nx_graph, dim=2)

            final_spring = {k:v.tolist() for k, v in final_spring.items()}
            nx_graph = nx.node_link_data(nx_graph)

            self.adjusted_nx_graph = nx_graph
            self.nx_graph = nx_graph
            self.final_spring = final_spring

            if self.timing:
                print('GENERATE NX GRAPH: ' + str(time.time() - start_time))

            return (nx_graph, final_spring)
        
        elif self.layout == 'spring':
            nx_graph = nx.from_pandas_edgelist(temp_df, 'source_id', 'target_id')

            final_spring = nx.spring_layout(nx_graph, dim=2, k=self.source_spread, iterations=self.simulation_iterations)

            final_spring = {k:v.tolist() for k, v in final_spring.items()}
            nx_graph = nx.node_link_data(nx_graph)

            self.adjusted_nx_graph = nx_graph
            self.nx_graph = nx_graph
            self.final_spring = final_spring

            if self.timing:
                print('GENERATE NX GRAPH: ' + str(time.time() - start_time))

            return (nx_graph, final_spring)

        elif self.layout == 'circular':
            nx_graph = nx.from_pandas_edgelist(temp_df, 'source_id', 'target_id')

            final_spring = nx.circular_layout(nx_graph, dim=2)

            final_spring = {k:v.tolist() for k, v in final_spring.items()}
            nx_graph = nx.node_link_data(nx_graph)

            self.adjusted_nx_graph = nx_graph
            self.nx_graph = nx_graph
            self.final_spring = final_spring

            if self.timing:
                print('GENERATE NX GRAPH: ' + str(time.time() - start_time))

            return (nx_graph, final_spring)

        elif self.layout == 'kk':
            nx_graph = nx.from_pandas_edgelist(temp_df, 'source_id', 'target_id')

            final_spring = nx.kamada_kawai_layout(nx_graph, dim=2)

            final_spring = {k:v.tolist() for k, v in final_spring.items()}
            nx_graph = nx.node_link_data(nx_graph)

            self.adjusted_nx_graph = nx_graph
            self.nx_graph = nx_graph
            self.final_spring = final_spring

            if self.timing:
                print('GENERATE NX GRAPH: ' + str(time.time() - start_time))

            return (nx_graph, final_spring)

        else:
            temp_df['edge_weight'] = temp_df.apply(lambda x: self._map_edge_weights(x['edge_value'], 2, 5), axis=1)
            
            nx_graph = nx.from_pandas_edgelist(temp_df, 'source_id', 'target_id', edge_attr=['edge_weight'])

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
                final_spring = nx.spring_layout(nx_graph, dim=2, weight='edge_weight', pos=pos_dict, fixed=fixed_list, k=self.source_spread, iterations=self.simulation_iterations)
            else:
                final_spring = nx.spring_layout(nx_graph, dim=2, weight='edge_weight', k=0.05, iterations=10)

            nx_graph = nx.node_link_data(nx_graph)
            final_spring = {k:v.tolist() for k, v in final_spring.items()}

            self.adjusted_nx_graph =  nx_graph
            self.nx_graph =  nx_graph
            self.final_spring = final_spring

            if self.timing:
                print('GENERATE NX GRAPH: ' + str(time.time() - start_time))

            return (nx_graph, final_spring)

    def _map_edge_weights(self, edge_val_column, min_weight, max_weight):
        """
        Maps edge values to weight values between min_weight and max_weight. These 
        weight values affect the edge "pull" experienced when simulating the graph.

        Args:
            edge_val_column (Pandas series): Pandas column (series) of values to be mapped.
            min_weight (float): Minimum desired weight value.
            max_weight (float): Maximum desired weight value.

        """

        edge_range = self.max_edge_value - self.min_edge_value

        if edge_range <= 0:
            return min_weight

        else:
            return min_weight + max_weight * ((edge_val_column - self.min_edge_value) / edge_range)

    def _generate_graph_elements(self):
        """
        Combines spacing information (from _generate_nx_graphs()), color information, etc. into a single elements 
        list that is directly fed into the dash cytoscape object.

        """

        start_time = time.time()

        nx_graph = nx.node_link_graph(self.nx_graph)
        elements = []
        
        # Items are type casted for easier communication between Dash callbacks w/ json.
        for node in nx_graph.nodes:
            if node in self.unique_target_nodes:
                node_value = 'None'
                size_val = self._generate_size(self.target_size)

            else:
                node_value = float(self.source_id_combined_scores_dict[node])
                size_val = self._generate_size(node_value)
                
            elements.append({
                'data': {
                    'id': str(node), 
                    'label': str(self.unique_id_data_dict[node]['name']), 
                    'value': node_value,
                    'type': str(self.unique_id_data_dict[node]['type']), 
                    'size': float(size_val), 
                    'label_size': float(size_val * 0.3),
                    'color': str(self.type_color_dict[self.unique_id_data_dict[node]['type']]),
                    'sn_or_tn': str(self.unique_id_data_dict[node]['sn_or_tn'])
                }, 
                'position': {'x': float(100 * self.final_spring[node][0]), 'y': float(100 * self.final_spring[node][1])}
            })

            if self.unique_id_data_dict[node]['sn_or_tn'] == 'source_node':
                if nx_graph.degree(node) != 0:
                    for target in nx_graph.neighbors(node):
                        edge_val = np.round(self.source_target_edge_value_dict[node][target], self.round_precision)
                        elements.append({
                            'data': {
                                'source': str(node), 
                                'target': str(target), 
                                'label': str(edge_val),
                                'size': float(self._generate_size(edge_val, node=False)), 
                                'label_size': float(self._generate_size(edge_val, node=False) * 10)
                            }
                        })

        self.elements = elements

        if self.timing:
            print('GENERATE ELEMENTS: ' + str(time.time() - start_time))

        return elements

    def _generate_table(self):
        """
        Creates a table with a similar layout as the data input based on filtered edges dataframe.

        """

        start_time = time.time()

        table_df = pd.read_json(self.edges_json).copy()

        table_df['combined_source_value'] = table_df['source_id']

        table_df = table_df.replace({"combined_source_value": self.source_id_combined_scores_dict})

        table_df = table_df.round(3)
        
        self.table_data = table_df
        self.data_table_columns = [{"name": i, "id": i} for i in table_df.columns]
        self.table_data = table_df.to_dict('records')

        if self.timing:
            print('GENERATE TABLE: ' + str(time.time() - start_time))

        return self.table_data

    def generate_node_data(self, selected_nodes_list):
        """
        Creates a node-specific layout for displaying individual node data in the app.

        """

        start_time = time.time()
        
        nx_graph = nx.node_link_graph(self.nx_graph)

        formatted_data_list = []

        for node in selected_nodes_list:
            if node['sn_or_tn'] == 'source_node':

                edges = {}
                for _, connecting_node in enumerate(nx_graph[node['id']]):
                    edges[str(self.unique_id_data_dict[connecting_node]['name']) + ' (ID:' + str(connecting_node) + ')'] = float(np.round(self.source_target_edge_value_dict[node['id']][connecting_node], self.round_precision))

                edges_sorted = dict(sorted(edges.items(), key=lambda item: item[1], reverse=True))

                data_dump = {
                    'source_id': node['id'], 
                    'source_name': node['label'], 
                    'node_type': node['type'], 
                    'combined_value': float(np.round(self.source_id_combined_scores_dict[node['id']], self.round_precision)), 
                    'sn_or_tn': 'source_node', 
                    'edges': edges_sorted
                    }

                formatted_data_list.append(json.dumps(data_dump, indent=2))
            
            if node['sn_or_tn'] == 'target_node':

                edges = {}
                for _, connecting_node in enumerate(nx_graph[node['id']]):
                    edges[str(self.unique_id_data_dict[connecting_node]['name']) + ' (ID:' + str(connecting_node) + ')'] = float(np.round(self.source_target_edge_value_dict[connecting_node][node['id']], self.round_precision))
                
                edges_sorted = dict(sorted(edges.items(), key=lambda item: item[1], reverse=True))

                data_dump = {
                    'node_id': node['id'], 
                    'node_name': node['label'], 
                    'sn_or_tn': 'target_node', 
                    'edges': edges_sorted
                    }

                formatted_data_list.append(json.dumps(data_dump, indent=2))

        if self.timing:
            print('NODE ELEMENTS: ' + str(time.time() - start_time))
            
        return formatted_data_list

    def update_graph_elements(self):
        """
        Updates the elements list based on graph filters/adjustments.

        """

        self.graph_update_method_timing_list = []

        start_time = time.time()
        if not self.graph_update_shortcut:
            self._filter_data()
        self.graph_update_method_timing_list.append(time.time() - start_time)

        start_time = time.time()
        self._trim_graph()
        self.graph_update_method_timing_list.append(time.time() - start_time)

        start_time = time.time()
        if not self.graph_update_shortcut:
            self._generate_table()
        self.graph_update_method_timing_list.append(time.time() - start_time)

        start_time = time.time()
        self._generate_color_mapping()
        self.graph_update_method_timing_list.append(time.time() - start_time)

        start_time = time.time()
        self._generate_graph_elements()
        self.graph_update_method_timing_list.append(time.time() - start_time)

        self.graph_update_shortcut = True

        return self.elements        

    def reset_graph(self):
        """
        Resets graph (and all sliders) based on the state that the data was initially loaded in. By not 
        re-simulating the graph, runtime is reduced substantially.

        """

        self.combined_value_range = self.combined_value_range_initial
        self.combined_value_bound = self.combined_value_bound_initial
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

        self.edges_json = self.edges_json_initial

        self.type_color_dict = self.type_color_dict_initial

        self.source_id_combined_scores_dict = self.source_id_combined_scores_dict_initial
        self.unique_target_nodes = self.unique_target_nodes_initial

        self.table_data = self.table_data_initial

        self.nx_graph = self.nx_graph_initial
        self.final_spring = self.nx_spring_initial
        self.adjusted_nx_graph = self.nx_graph_initial

    def load_additional_data(self, df_input):
        """
        Generates graph and slider values based on newly uploaded data.

        """
        
        self._initialize_data(df_input)

        self._initialize_graph_state()

        self._filter_data()

        self.type_color_dict_initial = self._generate_color_mapping()

        # Allows for simulation iterations to be set BEFORE data upload, potentially increasing load performance
        self.simulation_iterations_initial = self.simulation_iterations

        self.nx_graph_initial, self.nx_spring_initial = self._generate_nx_graphs()
        
        self.starting_elements = self._generate_graph_elements()

        self.table_data_initial = self._generate_table()

        return self.elements

    def simulate(self):
        """
        Essentially a _generate_nx_graphs() method that allows the user to re-simulate the graph if desired.

        """

        self._generate_nx_graphs()

        self.adjusted_nx_graph = self.nx_graph

        self._generate_graph_elements()

    def convert_to_json(self):
        
        return json.dumps(self.__dict__)