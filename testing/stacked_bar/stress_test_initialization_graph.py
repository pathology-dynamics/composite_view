import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 22}

matplotlib.rc('font', **font)

stacked_df = pd.read_csv('initialization_runtime_bar.csv', index_col=0)

print(stacked_df)

labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

initialize_data = stacked_df['initialize_data'].to_numpy()
initialize_simulation_iterations = stacked_df['initialize_simulation_iterations'].to_numpy()
initialize_graph_state = stacked_df['initialize_graph_state'].to_numpy()
color_mapping = stacked_df['color_mapping'].to_numpy()
initialize_nx_graph = stacked_df['initialize_nx_graph'].to_numpy()
generate_starting_elements = stacked_df['generate_starting_elements'].to_numpy()
generate_table = stacked_df['generate_table'].to_numpy()

width = 0.35

fig, ax = plt.subplots()

ax.bar(labels, initialize_nx_graph, width, label='generate_nx_graphs', color='#FF7F0E')
ax.bar(labels, initialize_graph_state, width, bottom=initialize_nx_graph, label='initialize_graph_state', color='#1F77B4')
ax.bar(labels, generate_table, width, bottom=initialize_nx_graph + initialize_graph_state, label='generate_table', color='#D62728')
ax.bar(labels, generate_starting_elements, width, bottom=initialize_nx_graph + initialize_graph_state + generate_table, label='generate_graph_elements', color='#2CA02C')

ax.set_ylabel('Runtime [s]')
ax.set_xlabel('Thousands of Source Nodes')
ax.set_title('Graph Initialization Runtime Broken Down by Method')

ax.legend()

plt.show()