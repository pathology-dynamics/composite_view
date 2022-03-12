import pandas as pd
import pickle as pkl
import numpy as np

df = pd.read_csv('simple_visualization_1.csv', index_col=0)

df.to_json()

infile = open('bug.pkl', 'rb')
bug_dict = pkl.load(infile)

import json

for i in bug_dict:
    for j in i:
        for k in i[j]:
            print(k)
            print(i[j][k])
            print(type(i[j][k]))