import pandas as pd
import numpy as np

from visualizer.generate_graph import Generate_Graph

edges_df_import = pd.read_csv('simple_visualization.csv', index_col=0)

print(edges_df_import)

grpah = Generate_Graph(edges_df=pd.read_csv('simple_visualization.csv'))