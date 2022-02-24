# Combined Scores Visualizer

This software is designed to visualize data, specifically combined scores data, utilizing graph theory and network analysis concepts. It was initially an offshoot of the literature-based discovery (LBD) tool SemNet [cite SemNet], where interpreting results (especially multi-target results) all but require such a tool. Given the nature of the SemNet results data, the visualizer was generalized to allow similar data sets to be used.

### Recomended Installation

1. Create and activate a custom virtual environment using the venv module. 
    * Documentation: https://docs.python.org/3/library/venv.html
    * Additional resource: https://python.land/virtual-environments/virtualenv
2. Once the virtual environment is activated, use pip to install all the necessary packages. Use the command `pip install path/to/visualizer/directory`, where the directory pointed to contains `setup.py`. Make sure the virtual environment is activated before using pip! 
    * Pip documentation: https://pypi.org/project/pip/
    * Potentially useful Stack thread: https://stackoverflow.com/questions/41535915/python-pip-install-from-local-dir
3. Simply run `visualizer.py` using the virtual environment described above and voilà! The app should be running on `localhost`, using whatever port/host is indicated in the output.
    * If you want to change the port, this thread should help you: https://stackoverflow.com/questions/45807913/plotly-dash-change-default-port.

### Useful Definitions

* **Combined score**: An aggregate score across multiple performance metrics. Take the Human Development Index (HDI) as an example (https://hdr.undp.org/en/indicators/137506). The HDI value assigned to a country is the geometric mean of three normalized performance metrics: life expectancy index, education index, and income index. The HDI value in this case is the combined score of the three performance metrics mentioned.

* **Source node**: A node that has an in-degree of zero. This term is borrowed from graph theory, and it is used to model the individual datapoints that are being visualized. Each source node represents a different entity, in this case a specific datapoint being compared to some number of target nodes. Continuing the HDI example, both Norway and Malawi, countries ranked based on their respective HDI, would each be represented in the visualizer as a source node.

* **Target node**: A node that has an out-degree of zero. This is another term borrowed from graph theory, and it is used to model the individual metrics used to form the combined score. In the HDI example, the target nodes would model life expectancy index, education index, and income index. In this example, each source node, representing each country, is assigned a value in respect to each target node.

* **Edge value**: The value assigned to each connection between a source and target node. In the HDI example, Norway is assigned a value in respect to all three target nodes. This relationship could be described as "Norway scores X in respect to life expectancy index" or "Malawi scores Y in respect to income index." In these example relationships, the X and Y values represent the edge values. These edge values are combined, per unique source node, to generate the combined score (which is then assigned to that source node).

* **Node Type**: A discrete categorization that can be assigned to both source and target nodes. For example, the source node "Norway" in the HDI example might be assigned the type "very_high_development" to indicate what development category the country resides in. The target node "life expectancy" might be assigned the type "target_node" to indicate that it's a target node. 

![](images/explanatory_graph_labeled.png)
Figure 1: A labeled example graph.
1. Source node
2. Target node
3. Edge value
4. Node type (color differences represent HDI bracket)

### Formatting the Data

The following table shows the data format required for the Visualizer. In Figure 1 above, the source node with the name "Sweden" is isolated, so it will act as the example. As seen in the image, Sweden is connected to the three targets: income_index, life_expectancy_index, and education_index. Each of these three connections is represented by the three rows in the table below.

|source_id|source_name|source_type|target_id|target_name|target_id|edge_value|
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
|Sweden|Sweden|very_high_development|income_index|income_index|target_node|0.952|
|Sweden|Sweden|very_high_development|life_expectancy_index|life_expectancy_index|target_node|0.966|
|Sweden|Sweden|very_high_development|education_index|education_index|target_node|0.917|

* **source_id**: A unique source node identifier. Unlike a source_name value, which could be shared between different nodes, the source_id value is unique to every rendered node. This is the value that the NetworkX graph object utilizes when buiding/simulating the graph.

* **source_name**: The name associated with a particular source node. As mentioned above, the source_name value could be shared between source nodes. 

* **source_type**: The "category" or "bucket" for the associated source node. In the example above, the category for Sweden is very_high_development.

* **target_id**: Unique target node identifier, similar to source_id.

* **target_name**: The name associated with a particular target node, similar to source_name.

* **target_type**: The "category" or "bucket" for the associated target node, similar to source_type.

* **edge_value**: The determined value or "metric" between the source node and target node. In the above table, each row indicates a unique connection between the row's source and target nodes. That connection is quantified by edge_value. As a rule of thumb, this value should share similar exist with the same units or on the same scale as other edge values (to keep certain sliders within the visualizer relevant).

As a brief aside, the table above shows all three HDI metrics for Sweden, modeled as a network. When calculating HDI, the geometric mean of all three edge values determines the actual HDI score, which is assigned to the country (this allows the country to be ranked in relation to other countries). The node size is determined by this combined value, which the visualizer will automatically calculate. The method for combining these scores can be directly modified within the `generate_graph` class, specifically the `_combine_values()` method. Other features, such as node size scaling, edge size scaling, etc. can also be adjusted.

### Filters/Interactive Elements

The Visualizer contains filters/interactive elements that can be accessed through the "Expand Settings" button. This will activate a dropdown with the following categories:

* **Graph Sliders**: Sliding elements that allow the user to set precise value bounds.
    * **Combined Value**: Set the combined value bound.
    * **Edge Value**: Set the edge value bound.
    * **Max Node Count**: Set the maximum number of source nodes that are rendered, sorted by combined value.

* **Node Filtering**: Dropdown elements that allow for certain nodes or types to be filtered in or out of the graph.
    * **Select Target Node Name(s)**: Dropdown element that allows specific target nodes to be shown.
    * **Select Source Node Name(s)**: Dropdown element that allows specific source nodes to be shown.
    * **Select Type(s)**: Dropdown element that allows for specific node types to be shown. Note: target types are included in the filter, so in order to see both sources and targets, both source and target types must be selected.

* **Graph Manipulation**: Varying elements used to improve the graph's visuals.
    * **Target Spread**: Slider element that adjusts the target node spread. This value is used in the NetworkX spring_layout method, which positions nodes using the Fruchterman-Reingold (FR) force-directed algorithm, by adjusting the optimal distance between nodes (k-value). The target spread affects the k-value for the first of two simulations, where the target nodes are placed and fixed (more on that later). 
    * **Source Spread**: Another sliding element. Source spread is essentially the same concept as the target spread, however the k-value being adjusted affects the second of the two simulations (where the source nodes fill in the space between targets). Adjusting this value will result in how densly packed the source nodes are.
    * **Node Size**: Sliding element that adjusts node size. The text associated with the node also scales. 
    * **Edge Size**: Sliding element that adjusts edge size. Like node size, the text associated with the edge also scales. 
    * **Simulation Iterations**: The number of iterations that the FR algorithm is run during the second simulation. This slider is a BIG performance contributer. If there are many thousands of data points being simulated, setting this to three or two, maybe even one or zero, would help performance significantly. Trimming the graph and setting value bounds also helps performance, allowing for more simulation iterations to be run (by reducing the number of simulated nodes). 
    * **Simulate**: Button element that re-simulates the graph based on the value set by Simulation Iterations (see details below).

* **Color Editing**: Varying elements used to change node colors.
    * **Type Gradient**: Two color selection elements that allow for the graph nodes to adopt a color gradient, based on the Colour package method range_to(). Generally, for this to work properly, colors should be adjacent on the color wheel.
    * **Selected Source Node Type Color**: A single color selection element that changes a selected node's type color. For example, if source_node_1 is of type type_1 and colored green, selecting that node and picking a new color using this element will change all type_1 nodes to that chosen color.
    * **Source Node Color**: Change the color of all source nodes.
    * **Target Node Color**: Change the color of all target nodes.
    * **Randomize Color**: Randomized all node type colors.
    
* **Node Data**: Data associated with a selected node or group of selected nodes (selecting multiple nodes can be achieved by shift-dragging). The node data presented includes node_id, node_name, node_type, combined_value (if the selected node is a source node), source_or_targe_node, and edge values (the names + id of connected nodes, along with the their respective edge_value).

* **Table Data**: Data table based on the trimming and filtering options applied to the graph. This table can be exported. 

Other interactive elements:

* **Reset Graph**: A button element that resets the graph to its original state. If no data has been uploaded, then the graph will revert to the initial load state. If data has been uploaded, the graph will revert to the state where the uploaded data was first initialized.

* **Upload Data**: A button element that allows for new data to be uploaded.
    * VERY IMPORTANT: The upload must follow the data format as described above and be a .csv file.

* **Download Image**: A button elements that downloads an image of the graph.

### Miscellaneous

* Performance: The simulation iterations sliding element is your best friend. Use it to increase performance (at the cost of potentially less graph interpretability).

* Simulation details.

    * Force-directed graph drawing algorithms, including the one used in this visualizer, are a class of algorithms with the purpose of reducing edge overlap and generating equal length edges. The algorithm used in the visualizer is one of these force-directed graph drawing algorithms, specifically the Fruchterman-Reingold force-directed algorithm (NetworkX spring_layout documentation: https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html).

    * **The problem**: Often, using algorithms such as FR isn't enough. In the case of the Visualizer, what would often happen (without any tampering) is that nodes would often get "stuck" and fail to reach their (often obvious) optimal positions. All of the source nodes would cluster with target nodes in the center, creating a very obvious visibility problem (see below).

    * **The solution**: Given the structure of the data, generally there will be few highly connected target nodes and many slightly connected source nodes. These source nodes will cluster together based on which shared targets they are connected to; spacing the target nodes and source node "clusters" in an intuitive way is the goal. Given this feature of the data, a method has been devised to initialize the nodes and prevent this "sticking" behavior.

        * **Simulation 1**: The first step is simulating all of the target nodes WITHOUT the source nodes. The FR algorithm is used, and targets have shared edges IF they share a common source node (target nodes connected to the same source node "clusters" will share edges). The more source nodes shared between sets of targets, the lower the edge weights between these targets (edge weight increases the "spring" force of the edges in the FR algorithm). By setting shared source nodes to be inversely proportional to edge weight, more "room" is allowed between targets that have many shared connected source nodes.

        * **Simulation 2**: Once the FR algorithm is run given the setup stated above, the source nodes are "filled in." Each group of source nodes that share common targets are placed at the centroid of this set of shared target nodes. Without additional modification, these source nodes would simply not move after being placed (they occupy the same position, therefore there is no "repelling" force), so each source node is placed around the centroid based on a Gaussian distribution. This distribution allows the source nodes enough room to interact and naturally repell. This step is where setting simulation iterations to a lower value can vastly improve performance; force-directed algorithms can be computationally expensive and slow, so reducing algorithm iterations generally improves performance. 

            * As a final aside, each source-target edge is assigned a weight value based on the edge value associated with the two nodes. This weight will "pull" the source node harder in the direction of the target(s) it is most highly connected to. 

* Shift-dragging allows multiple source/target nodes to be selected. If the algorithm mentioned above doesn't produce optimal spacing results, this is the next best bet... or just re-simulate the graph.

![](images/spring_layout.png)
Figure 2: The graph visualized using a spring layout.

![](images/adjusted_spring_layout.png)
Figure 3: The graph visualized using the adjusted spring layout, as described above. The target nodes are much easier to identify, and the source node clusters can be easily parsed.

## Contact
If you have any questions regard the Visualizer, email sallegri3@gatech.edu.
