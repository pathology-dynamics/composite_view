import pandas as pd
import os

def compositeview_data_from_semnet(path=str, generate_csv=True):
    """
    SemNet specific function to combine multiple SemNet outputs into a single, formatted DataFrame/CSV.

    Args:
        path (str): Path to top level directory, where each SemNet output (including multiple targets)
                    is placed into a sub-directory within the path directory.
        generate_csv (bool): Bool to generate CSV file for CompositeView use.

    Returns:
        pandas DataFrame: Formatted DataFrame for use within CompositeView.

    """

    df_list = []
    semnet_run_count = 0

    for root, _, files in os.walk(path, topdown=True):
        for name in files:
            file = os.path.join(root, name)

            if 'csv' in file:
                df = pd.read_csv(file, index_col=0)
                df['source_node'] = df['source_node'].astype(str) + '_' + str(semnet_run_count)
                df_list.append(df)
        
        semnet_run_count = semnet_run_count + 1

    combined_df = pd.concat(df_list, ignore_index=True)

    data = {
        'source_id': combined_df['source_node'], 
        'source_name': combined_df['source_name'], 
        'source_type': combined_df['source_type'], 
        'target_id': combined_df['target_node'], 
        'target_name': combined_df['target_name'], 
        'target_type': 'target_node', 
        'edge_value': combined_df['hetesim_score']
    }

    edges_df = pd.DataFrame(data)

    if generate_csv:
        edges_df.to_csv('formatted_cv_data.csv')

    return edges_df