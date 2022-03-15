import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl

font = {'size': 22}

matplotlib.rc('font', **font)

with open('graph_initialization_timing.pkl', 'rb') as f:
    initialization_timing = pkl.load(f)

with open('attribute_load_timing.pkl', 'rb') as f:
    attribute_load_timing = pkl.load(f)

with open('graph_update_timing.pkl', 'rb') as f:
    update_timing = pkl.load(f)

with open('app_startup_timing.pkl', 'rb') as f:
    app_timing = pkl.load(f)

print(len(app_timing))
x = np.arange(20, (20 + len(app_timing) * 20), 20)

#plt.scatter(x, simulation_timing, linestyle='--')
plt.plot(x, initialization_timing, linestyle='-', label='Graph Initialization')
plt.plot(x, update_timing, linestyle='-', label='Graph Update')
plt.plot(x, attribute_load_timing, linestyle='-', label='Attribute Loading')
plt.plot(x, app_timing, linestyle='-', label='Application Layout and Callback Creation')

plt.xlabel('Source Node Count')
plt.ylabel('Runtime [s]')
plt.title('Visualizer Runtime Analysis: Source Node Count vs. Runtime')

plt.legend()
plt.show()