import numpy as np
import pandas as pd
import networkx as nx
import itertools
import networkx as nx
import json
from colour import Color

class Generate_Graph:
    def __init__(self, df_list=None, target_cui_target_name_dict=None) -> None:

        self._format_data(df_list, target_cui_target_name_dict)
        self._initialize_starting_elements()
        self._adjust_data()

        self._generate_color_mapping()
        self.starting_table_data = self._generate_table()

        self.starting_nx_graph = self._generate_nx_graph()
        self.starting_elements = self.generate_graph_elements()

    def reset_graph(self):

        self.randomized_color_button_clicks = self.randomized_color_button_clicks_initial

        self.node_hetesim_range = self.node_hetesim_range_initial
        self.edge_hetesim_range = self.edge_hetesim_range_initial
        self.max_node_count = self.max_node_count_initial

        self.specific_target_dropdown = self.specific_target_dropdown_initial
        self.specific_source_dropdown = self.specific_source_dropdown_initial
        self.specific_type_dropdown = self.specific_type_dropdown_initial

        self.target_spread = self.target_spread_initial
        self.sn_spread = self.sn_spread_initial

        self.gradient_start = self.gradient_start_initial
        self.gradient_end = self.gradient_end_initial
        self.selected_type_color = self.selected_type_color_initial
        self.target_color = self.target_color_initial

    def _generate_dummy_graph(self):

        base_names = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        relationship_list = [{'T1', 'T2'}, {'T2', 'T3'}, {'T2', 'T3', 'T4'}, {'T4', 'T5'}, {'T5', 'T6'}, {'T6', 'T7'}, {'T7', 'T8'}, {'T7', 'T4'}]
        type_list = ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'type_6', 'type_7', 'type_8']
        dummy_df_list = []

        for i in range(len(relationship_list)):
            sn_list = []
            sn_name_list = []
            target_list = []
            hetesim_list = []
            type_list_random = []

            for name in base_names:
                sn_list.append(name)
                sn_name_list.append(name + '_name')

                temp_dict = {}
                for j in relationship_list[i]:
                    temp_random_val = np.abs(np.random.normal(0, 0.5))

                    if temp_random_val > 1:
                        temp_random_val = 1
                    if temp_random_val < 0:
                        temp_random_val = 0

                    temp_dict[j] = temp_random_val
                    
                target_list.append(temp_dict)
                type_list_random.append(type_list[np.random.randint(0, len(type_list))])

            data = {
                'sn': sn_list, # CUI
                'sn_name': sn_name_list, # SN name
                'targets': target_list, # Target CUI
                'type': type_list_random # SN type
                }

            dummy_df_list.append(pd.DataFrame(data))

        dummy_target_cui_target_name_dict = {
            'T1': 'T1_name', 
            'T2': 'T2_name', 
            'T3': 'T3_name', 
            'T4': 'T4_name', 
            'T5': 'T5_name', 
            'T6': 'T6_name', 
            'T7': 'T7_name', 
            'T8': 'T8_name'
            }

        return [dummy_df_list, dummy_target_cui_target_name_dict]

    def _format_data(self, df_list_input, target_cui_target_name_dict_input):

        if (df_list_input == None) & (target_cui_target_name_dict_input == None):
            dummy_data = self._generate_dummy_graph()
            df_list_input = dummy_data[0]
            target_cui_target_name_dict_input = dummy_data[1]

        formatted_df_list = []

        target_relationship_list_formatted = []
        sn_mean_hetesim_dict_formatted = {}
        sn_type_dict_formatted = {}
        sn_cui_sn_name_dict = {}

        for i, df in enumerate(df_list_input):
            sub_df_formatted = pd.DataFrame(columns=['sn', 'type', 'sn_name', 'target', 'hetesim', 'mean_hetesim'])

            for _, row in df.iterrows():
                sn_adjusted = str(row['sn']) + '_' + str(i)
                sn_type_adjusted = row['type']
                sn_name_adjusted = row['sn_name']
                sn_mean_hetesim_adjusted = []

                target_dict = row['targets']

                for target in target_dict:
                    sn_mean_hetesim_adjusted.append(target_dict[target])
                
                for target in target_dict:
                    sub_df_formatted = sub_df_formatted.append({
                        'sn': sn_adjusted,
                        'type': sn_type_adjusted,
                        'sn_cui': row['sn'],
                        'sn_name': sn_name_adjusted,
                        'target': target, 
                        'hetesim': target_dict[target],
                        'mean_hetesim': np.mean(sn_mean_hetesim_adjusted)
                        }, ignore_index=True)

                    sub_df_formatted = sub_df_formatted.sort_values(by=['mean_hetesim'], ascending=False)

                sn_mean_hetesim_dict_formatted[sn_adjusted] = np.mean(sn_mean_hetesim_adjusted)
                sn_type_dict_formatted[sn_adjusted] = sn_type_adjusted

                sn_cui_sn_name_dict[row['sn']] = sn_name_adjusted

            target_relationship_list_formatted.append(tuple(df.iloc[0]['targets'].keys()))

            formatted_df_list.append(sub_df_formatted)

        self.formatted_df_list = formatted_df_list
        self.target_relationship_list_formatted = target_relationship_list_formatted
        self.sn_mean_hetesim_dict_formatted = sn_mean_hetesim_dict_formatted
        self.sn_type_dict_formatted = sn_type_dict_formatted

        self.target_cui_target_name_dict = target_cui_target_name_dict_input
        self.sn_cui_sn_name_dict = sn_cui_sn_name_dict

        self.combined_formatted_df = pd.concat(formatted_df_list)
        self.unique_types = sorted(self.combined_formatted_df['type'].unique())

        # self.adjusted_df_list = self.formatted_df_list.copy()

        return formatted_df_list

    def _initialize_starting_elements(self):

        # Node HeteSim slider value initialization
        self.node_hetesim_range = [
            np.round(self.combined_formatted_df['mean_hetesim'].min(), 3), 
            np.round(self.combined_formatted_df['mean_hetesim'].max(), 3)
            ]

        self.node_hetesim_range_initial = self.node_hetesim_range
        
        self.node_hetesim_step_size = np.round((self.node_hetesim_range[1] - self.node_hetesim_range[0]) / 100, 3)

        self.node_hetesim_range_start = self.node_hetesim_range[0] - self.node_hetesim_step_size
        if self.node_hetesim_range_start < 0:
            self.node_hetesim_range_start = 0

        self.node_hetesim_range_end = self.node_hetesim_range[1] + self.node_hetesim_step_size
        if self.node_hetesim_range_end > 1:
            self.node_hetesim_range_end = 1

        # Edge HeteSim slider value initialization
        min_edge_value = np.inf
        for df in self.formatted_df_list:
            if df['hetesim'].min() < min_edge_value:
                min_edge_value = df['hetesim'].min()
        self.min_edge_value = min_edge_value

        max_edge_value = -np.inf
        for df in self.formatted_df_list:
            if df['hetesim'].max() > max_edge_value:
                max_edge_value = df['hetesim'].max()
        self.max_edge_value = max_edge_value

        self.edge_hetesim_range = [
            np.round(self.min_edge_value, 3), 
            np.round(self.max_edge_value, 3)
            ]

        self.edge_hetesim_range_initial = self.edge_hetesim_range
        
        self.edge_hetesim_step_size = np.round((self.max_edge_value - self.min_edge_value) / 100, 3)

        self.edge_hetesim_range_start = self.min_edge_value - self.edge_hetesim_step_size
        if self.edge_hetesim_range_start < 0:
            self.edge_hetesim_range_start = 0

        self.edge_hetesim_range_end = self.max_edge_value + self.edge_hetesim_step_size
        if self.edge_hetesim_range_end > 1:
            self.edge_hetesim_range_end = 1

        # Max node slider value initialization
        max_node_count = 0
        for df in self.formatted_df_list:
            if df.shape[0] > max_node_count:
                max_node_count = df.shape[0]
        self.max_node_count = int(max_node_count / 2)

        self.max_node_count_initial = self.max_node_count

        self.max_node_size_ref = 10 / self.combined_formatted_df['mean_hetesim'].max()

        # Specific target filtering initialization
        self.specific_target_dropdown = []

        self.specific_target_input_options = []
        inverted_target_cui_target_name_dict = {v: k for k, v in self.target_cui_target_name_dict.items()}

        for entry in sorted([self.target_cui_target_name_dict.values()]):
            self.specific_target_input_options.append({'label': entry, 'value': inverted_target_cui_target_name_dict[entry]})

        self.specific_target_dropdown_initial = self.specific_target_dropdown

        # Specific source filtering initialization
        self.specific_source_dropdown = []

        self.specific_source_input_options = []
        inverted_sn_cui_sn_name_dict = {v: k for k, v in self.sn_cui_sn_name_dict.items()}

        for entry in sorted([self.sn_cui_sn_name_dict.values()]):
            self.specific_source_input_options.append({'label': entry, 'value': inverted_sn_cui_sn_name_dict[entry]})

        self.specific_source_input_options_initial = self.specific_source_input_options

        self.specific_source_dropdown_initial = self.specific_source_dropdown

        # Specific type filtering initialization
        self.specific_type_dropdown = []

        self.specific_type_input_options = []

        for entry in self.unique_types:
            self.specific_type_input_options.append({'label': entry, 'value': entry})

        self.specific_type_dropdown_initial = self.specific_type_dropdown

        # TN/SN spread initialization
        self.target_spread = 1
        self.target_spread_initial = self.target_spread
        self.sn_spread = 0.1
        self.sn_spread_initial = self.sn_spread

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

        # Target color initialization
        self.target_color = '#272B30'
        self.target_color_primacy = False
        self.target_color_initial = self.target_color

        # Randomized color initialization
        self.randomized_color_button_clicks = None
        self.random_color_primacy = True
        self.randomized_color_button_clicks_initial = self.randomized_color_button_clicks

        # Reset button
        self.reset_button_clicks = None
    
    def _adjust_data(self):

        adjusted_df_list = self.formatted_df_list.copy()

        if self.max_node_count != self.max_node_count_initial:
            adjusted_df_list = self._adjust_max_nodes(adjusted_df_list)
            self._adjust_dropdown_options(adjusted_df_list)

        if self.node_hetesim_range != self.node_hetesim_range_initial:
            adjusted_df_list = self._adjust_node_hetesim_range(adjusted_df_list)

        if self.edge_hetesim_range != self.edge_hetesim_range_initial:
            adjusted_df_list = self._adjust_edge_hetesim_range(adjusted_df_list)

        if self.specific_target_dropdown != self.specific_target_dropdown_initial:
            adjusted_df_list = self._adjust_specific_target_dropdown(adjusted_df_list)

        if self.specific_source_dropdown != self.specific_source_dropdown_initial:
            adjusted_df_list = self._adjust_specific_source_dropdown(adjusted_df_list)

        if self.specific_type_dropdown != self.specific_type_dropdown_initial:
            adjusted_df_list = self._adjust_specific_type_dropdown(adjusted_df_list)      
        
        self.adjusted_df_list = adjusted_df_list

        self._generate_table()

        self.specific_source_input_options = None

    def _generate_nx_graph(self):

        self.combined_adjusted_df = pd.concat(self.adjusted_df_list)

        target_edges = []
        for relationship in self.target_relationship_list_formatted:
            for subset in itertools.combinations(relationship, 2):
                target_edges.append(subset)

        initial_graph = nx.Graph(target_edges)
        initial_spring = nx.spring_layout(initial_graph, dim=2, k=self.target_spread, iterations=100)

        pos_dict = {}
        fixed_list = []

        for entry in initial_graph.nodes:
            pos_dict[entry] = [initial_spring[entry][0], initial_spring[entry][1]]
            fixed_list.append(entry)

        final_graph = nx.from_pandas_edgelist(self.combined_adjusted_df, 'sn', 'target', ['sn_name', 'sn_cui', 'hetesim', 'mean_hetesim'])
        final_spring_graph = nx.spring_layout(final_graph, dim=2, pos=pos_dict, fixed=fixed_list, k=self.sn_spread, iterations=100)

        for _, row in self.combined_adjusted_df.iterrows():
            final_graph.nodes[row['sn']]['id'] = row['sn']
            final_graph.nodes[row['sn']]['cui'] = row['sn_cui']
            final_graph.nodes[row['sn']]['name'] = row['sn_name']
            final_graph.nodes[row['sn']]['type'] = row['type']
            final_graph.nodes[row['sn']]['mean_hetesim'] = row['mean_hetesim']
            final_graph.nodes[row['sn']]['size'] = self.max_node_size_ref * row['mean_hetesim']
            final_graph.nodes[row['sn']]['color'] = str(self.type_color_dict[self.sn_type_dict_formatted[row['sn']]])
            final_graph.nodes[row['sn']]['sn_or_tn'] = 'source_node'
            final_graph.nodes[row['sn']]['position'] = {'x': 100 * final_spring_graph[row['sn']][0], 'y': 100 * final_spring_graph[row['sn']][1]}

        for target_cui in self.combined_adjusted_df['target'].unique():
            final_graph.nodes[target_cui]['id'] = target_cui
            final_graph.nodes[target_cui]['name'] = self.target_cui_target_name_dict[target_cui]
            final_graph.nodes[target_cui]['color'] = self.type_color_dict['target']
            final_graph.nodes[target_cui]['size'] = 10
            final_graph.nodes[target_cui]['sn_or_tn'] = 'target_node'
            final_graph.nodes[target_cui]['position'] = {'x': 100 * final_spring_graph[target_cui][0], 'y': 100 * final_spring_graph[target_cui][1]}

        self.nx_graph = final_graph

        return final_graph

    def generate_graph_elements(self):

        elements = []

        for node in self.nx_graph.nodes:
            if self.nx_graph.nodes[node]['sn_or_tn'] == 'target_node':
                elements.append({
                    'data': {
                        'id': self.nx_graph.nodes[node]['id'], 
                        'label': self.nx_graph.nodes[node]['name'], 
                        'size': self.nx_graph.nodes[node]['size'], 
                        'color': self.nx_graph.nodes[node]['color'],
                        'sn_or_tn': 'target_node'}, 
                    'position': self.nx_graph.nodes[node]['position']})

            if self.nx_graph.nodes[node]['sn_or_tn'] == 'source_node':
                elements.append({
                    'data': {
                        'id': self.nx_graph.nodes[node]['id'], 
                        'label': self.nx_graph.nodes[node]['name'],
                        'mean_hetesim': self.nx_graph.nodes[node]['mean_hetesim'], 
                        'size': self.nx_graph.nodes[node]['size'], 
                        'color': self.nx_graph.nodes[node]['color'],
                        'sn_or_tn': 'source_node'}, 
                    'position': self.nx_graph.nodes[node]['position']})

        for node_1 in self.nx_graph:
            for node_2 in self.nx_graph[node_1]:
                elements.append({
                    'data': {'source': node_1, 
                    'target': node_2, 
                    'size': np.round(self.nx_graph[node_1][node_2]['hetesim'], 2), 
                    'label': np.round(self.nx_graph[node_1][node_2]['hetesim'], 2)}})

        self.elements = elements

        return elements

    def update_graph_elements(self):
        
        self._generate_color_mapping()
        self._adjust_data()
        self._generate_nx_graph()
        self.generate_graph_elements()

        return self.elements

    def _generate_color_mapping(self):

        if self.random_color_primacy:
            self.type_color_dict = {}
            color_intervals = (330 / len(self.unique_types))
            random_color_list = []

            for i in range(len(self.unique_types)):

                normal_val = np.abs(np.random.normal(0.5, 0.33, 1)[0])
                if normal_val > 1 : normal_val = 1

                normal_color = color_intervals * normal_val
                random_color_list.append('hsl(' + str((color_intervals * i) + normal_color) + ', 100%, 60%)')

            random_color_list = np.random.choice(random_color_list, len(self.unique_types), replace=False)

            for i, type in enumerate(self.unique_types):
                self.type_color_dict[type] = random_color_list[i]

            self.type_color_dict['target'] = 'hsl(0, 100%, 60%)'

            self.random_color_primacy = False

        if self.gradient_color_primacy:
            starting_color = Color(self.gradient_start)
            color_gradient_list = list(starting_color.range_to(Color(self.gradient_end), len(self.unique_types)))

            for i, type in enumerate(self.unique_types):
                self.type_color_dict[type] = color_gradient_list[i]

            self.gradient_color_primacy = False

        if self.type_color_primacy:
            for type in self.selected_types:
                self.type_color_dict[type] = self.selected_type_color

            self.type_color_primacy = False

        if self.target_color_primacy:
            self.type_color_dict['target'] = self.target_color

            self.target_color_primacy = False

    def _adjust_node_hetesim_range(self, df_list):

        adjusted_df_list = []

        for df in df_list:
            df = df[df['mean_hetesim'] >= self.node_hetesim_range[0]]
            df = df[df['mean_hetesim'] <= self.node_hetesim_range[1]]
            adjusted_df_list.append(df)

        return adjusted_df_list

    def _adjust_edge_hetesim_range(self, df_list):

        adjusted_df_list = []

        for df in df_list:
            df = df[df['hetesim'] >= self.edge_hetesim_range[0]]
            df = df[df['hetesim'] <= self.edge_hetesim_range[1]]
            adjusted_df_list.append(df)

        return adjusted_df_list

    def _adjust_max_nodes(self, df_list):

        adjusted_df_list = []

        for df in df_list:
            unique_max_count_sns = df['sn'].unique()[:self.max_node_count]
            df = df[df['sn'].isin(unique_max_count_sns)]
            adjusted_df_list.append(df)

        return adjusted_df_list

    def _adjust_specific_target_dropdown(self, df_list):

        adjusted_df_list = []

        for df in df_list:
            adjusted_df_list.append(df[df['target'].isin(self.specific_target_dropdown)])

        return adjusted_df_list

    def _adjust_specific_source_dropdown(self, df_list):

        adjusted_df_list = []

        for df in df_list:
            adjusted_df_list.append(df[df['sn_cui'].isin(self.specific_source_dropdown)])

        return adjusted_df_list

    def _adjust_specific_type_dropdown(self, df_list):

        adjusted_df_list = []

        for df in df_list:
            adjusted_df_list.append(df[df['type'].isin(self.specific_type_dropdown)])

        return adjusted_df_list

    def _adjust_dropdown_options(self, df_list):
        specific_source_input_options_list = []
        remaining_sources = []

        for df in df_list:
            remaining_sources = remaining_sources + list(df['sn_cui'].unique())

        for entry in self.specific_source_input_options_initial:
            if entry['value'] in remaining_sources:
                specific_source_input_options_list.append({'label': entry['label'], 'value': entry['value']})

        self.specific_source_input_options = specific_source_input_options_list

    def generate_node_data(self, selected_nodes_list):

        formatted_data_list = []

        for node in selected_nodes_list:
            if node['sn_or_tn'] == 'source_node':
                self.selected_types.add(self.starting_nx_graph.nodes[node['id']]['type'])

                edges = {}
                for _, connecting_node in enumerate(self.starting_nx_graph[node['id']]):
                    edges[str(self.target_cui_target_name_dict[connecting_node]) + ' (CUI:' + str(connecting_node) + ')'] = float(np.round(self.starting_nx_graph[node['id']][connecting_node]['hetesim'], 3))

                edges_sorted = dict(sorted(edges.items(), key=lambda item: item[1], reverse=True))

                data_dump = {
                    'node_cui': self.starting_nx_graph.nodes[node['id']]['cui'], 
                    'node_name': self.starting_nx_graph.nodes[node['id']]['name'], 
                    'node_type': self.starting_nx_graph.nodes[node['id']]['type'], 
                    'mean_hetesim': float(np.round(self.starting_nx_graph.nodes[node['id']]['mean_hetesim'], 3)), 
                    'sn_or_tn': 'source_node', 
                    'edge_hetesim': edges_sorted
                    }

                formatted_data_list.append(json.dumps(data_dump, indent=2))

            if node['sn_or_tn'] == 'target_node':

                edges = {}
                for _, connecting_node in enumerate(self.starting_nx_graph[node['id']]):
                    edges[str(self.starting_nx_graph.nodes[connecting_node]['name']) + ' (CUI:' + str(self.starting_nx_graph.nodes[connecting_node]['cui']) + ')'] = float(np.round(self.starting_nx_graph[node['id']][connecting_node]['hetesim'], 3))
                
                edges_sorted = dict(sorted(edges.items(), key=lambda item: item[1], reverse=True))

                data_dump = {
                    'node_cui': self.starting_nx_graph.nodes[node['id']]['id'], 
                    'node_name': self.starting_nx_graph.nodes[node['id']]['name'], 
                    'sn_or_tn': 'target_node', 
                    'edge_hetesim': edges_sorted
                    }

                formatted_data_list.append(json.dumps(data_dump, indent=2))

        return formatted_data_list


    def _generate_table(self):

        combined_df_generating = pd.concat(self.adjusted_df_list)

        combined_df_generating = combined_df_generating.round({'hetesim': 3, 'mean_hetesim': 3})
        combined_df_generating = combined_df_generating[['sn_name', 'target', 'hetesim', 'mean_hetesim', 'type', 'sn_cui']]
        combined_df_generating = combined_df_generating.replace({"target": self.target_cui_target_name_dict})

        sorted_df_list = []

        for target in combined_df_generating['target'].unique():
            specific_target_df = combined_df_generating[combined_df_generating['target'] == target].copy()
            sorted_df_list.append(specific_target_df.sort_values(['hetesim'], ascending=False))

        combined_df_local = pd.concat(sorted_df_list)
        
        self.table_data = combined_df_local
        self.data_table_columns = [{"name": i, "id": i} for i in combined_df_local.columns]
        self.table_data = combined_df_local.to_dict('records')

        return self.table_data
