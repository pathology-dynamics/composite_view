import pandas as pd
import numpy as np

from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

from scipy.signal import argrelextrema

# Misc function for finding highly connected nodes given CompositeView data format
def identify_highly_connected_nodes(edges_df_input): 

    source_edge_series = edges_df_input['source_id']
    target_edge_series = edges_df_input['target_id']

    node_instances_df = pd.concat([source_edge_series, target_edge_series], ignore_index=True).to_frame(name='nodes')
    node_instances_df['edge_count'] = node_instances_df.groupby('nodes')['nodes'].transform('count')
    node_instances_df = node_instances_df.drop_duplicates(ignore_index=True)
    node_instances_df = node_instances_df.sort_values(by=['edge_count'], ascending=False)

    edge_count_array = node_instances_df['edge_count'].to_numpy()

    X_1 = edge_count_array.reshape(-1, 1)

    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(X_1)
    edge_domain = np.linspace(0, np.max(X_1), np.shape(X_1)[0])
    kde_curve = kde.score_samples(edge_domain.reshape(-1, 1))

    # plt.plot(edge_domain, kde_curve)
    # plt.show()

    min_array = argrelextrema(kde_curve, np.less)[0]

    range_list = [0]

    for i in list(min_array):
        range_list.append(edge_domain[i])

    range_list.append(np.max(X_1))
    range_list = range_list[::-1]

    for i in range(len(range_list) - 1):
        node_instances_df.loc[node_instances_df['edge_count'].between(range_list[i + 1], range_list[i]), 'cluster_1'] = i

    clustered_df = node_instances_df

    clustered_val_array = clustered_df['cluster_1'].to_numpy()

    X_2 = clustered_val_array.reshape(-1, 1)

    (unique, counts) = np.unique(X_2, return_counts=True)
    frequencies = np.asarray((unique, counts)).T

    freq_vals = frequencies[:, 1].reshape(-1, 1)

    kmeans = KMeans(n_clusters=2).fit(freq_vals)

    mapping_array = np.append(frequencies, kmeans.labels_.reshape(-1, 1), axis=1)
    mapping_array = np.delete(mapping_array, 1, 1)

    mapping_dict = {}
    for cluster_1, cluster_2 in mapping_array:
        mapping_dict[cluster_1] = cluster_2

    clustered_df['cluster_2'] = clustered_df['cluster_1'].map(mapping_dict)

    highly_connected_nodes = clustered_df[clustered_df['cluster_2'] == 0]['nodes'].to_list()
    slightly_connected_nodes = clustered_df[clustered_df['cluster_2'] == 1]['nodes'].to_list()

    return (highly_connected_nodes, slightly_connected_nodes)