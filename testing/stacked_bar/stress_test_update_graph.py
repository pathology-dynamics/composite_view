import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 22}

matplotlib.rc('font', **font)

stacked_df = pd.read_csv('update_runtime_bar.csv', index_col=0)

print(stacked_df)

labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

filter_data = stacked_df['filter_data'].to_numpy()
trim_graph = stacked_df['trim_graph'].to_numpy()
generate_table = stacked_df['generate_table'].to_numpy()
color_mapping = stacked_df['color_mapping'].to_numpy()
generate_graph_elements = stacked_df['generate_graph_elements'].to_numpy()

width = 0.35

fig, ax = plt.subplots()

ax.bar(labels, filter_data, width, label='filter_data', color='#9467BD')
ax.bar(labels, generate_table, width, bottom=filter_data, label='generate_table', color='#D62728')
ax.bar(labels, trim_graph, width, bottom=filter_data + generate_table, label='trim_graph', color='#17BECF')
ax.bar(labels, generate_graph_elements, width, bottom=filter_data + generate_table + trim_graph, label='generate_graph_elements', color='#2CA02C')

ax.set_ylabel('Runtime [s]')
ax.set_xlabel('Thousands of Source Nodes')
ax.set_title('Graph Update Runtime Broken Down by Method')
ax.legend()

plt.show()