import dash
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output
from dash import html, dcc, State
from dash import dash_table as dt

import pandas as pd

import base64
import time
import io

from generate_graph import Generate_Graph

# Set to True if you want timing data for graph startup/updates.
test_timing = False
if test_timing:
    start_time_global = time.time()
    print('================')

graph = Generate_Graph(timing=test_timing)

app = dash.Dash(external_stylesheets=[__name__, dbc.themes.SLATE])

class UI_Tracker:
    '''
    Class for tracking all UI elements that change throughout app run.

    '''

    def __init__(self):
        self.set_initial_state()

    def set_initial_state(self):
        '''
        Initializes all UI states that don't directly interact with the graph object.

        '''

        self.card_stack_tracking = []
        self.dropdown_cards = []

        self.settings_button_toggle = False
        self.settings_button_class = 'button_disabled'
        self.settings_button_text = 'Expand Settings'

        self.graph_sliders_button_toggle = False
        self.graph_sliders_button_class = 'button_disabled'

        self.node_filtering_button_toggle = False
        self.node_filtering_button_class = 'button_disabled'

        self.graph_manipulation_button_toggle = False
        self.graph_manipulation_button_class = 'button_disabled'

        self.color_editing_button_toggle = False
        self.color_editing_button_class = 'button_disabled'

        self.node_data_button_toggle = False
        self.node_data_button_class = 'button_disabled'

        self.table_data_button_toggle = False
        self.table_data_button_class = 'button_disabled'

        self.reset_button_clicks = None
        self.simulate_button_clicks = None
        self.randomized_color_button_clicks = None

        self.color_change_only = True
        self.graph_manipulation_only = True

        self.download_graph_clicks = None

        self.display_gradient_start_color = graph.gradient_start_initial
        self.display_gradient_end_color = graph.gradient_end_initial
        self.display_selected_type_color = graph.selected_type_color_initial
        self.display_source_color = graph.source_color_initial
        self.display_target_color = graph.target_color_initial

ui_tracker = UI_Tracker()

# DBC card that defines graph slider layout.
graph_sliders = dbc.Card(
    dbc.CardBody(
        [
            html.H6('Combined Value:', className='card-text', style={'marginBottom': '3%'}),
            html.Div(
                dcc.RangeSlider(
                    id='combined_value_range_slider',
                    min=graph.combined_value_bound[0],
                    max=graph.combined_value_bound[1],
                    step=graph.combined_value_step_size,
                    value=[
                        graph.combined_value_range_initial[0], 
                        graph.combined_value_range_initial[1]
                        ],
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ),
                style={'marginBottom': '6%'}
            ),

            html.H6('Edge Value:', className='card-text', style={'marginBottom': '3%'}),

            html.Div(
                dcc.RangeSlider(
                    id='edge_value_range_slider',
                    min=graph.edge_value_bound[0],
                    max=graph.edge_value_bound[1],
                    step=graph.edge_value_step_size,
                    value=[
                        graph.edge_value_range_initial[0],
                        graph.edge_value_range_initial[1]
                        ],
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ), 
                style={'marginBottom': '6%'} 
            ),

            html.H6('Max Node Count:', className='card-text', style={'marginBottom': '3%'}),

            html.Div(
                dcc.Slider(
                    id='max_node_slider',
                    min=1,
                    max=graph.max_node_count_initial,
                    step=1,
                    value=graph.max_node_count_initial,
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ), 
                style={'marginBottom': '6%'} 
            )
        ]
    )
)

# DBC card that defines graph filter layout.
node_filtering = dbc.Card(
    dbc.CardBody(
        [
            html.Div(children=[
                html.H6('Select Target Node Name(s):', className='card-text'),
                dcc.Dropdown(
                    id='specific_target_dropdown', 
                    options=graph.target_dropdown_options_initial,
                    value=[],
                    multi=True)
                ], 
                style={'marginBottom': '6%'}
            ), 

            html.Div(children=[
                html.H6('Select Source Node Name(s):', className='card-text'),
                dcc.Dropdown(
                    id='specific_source_dropdown', 
                    options=graph.source_dropdown_options_initial,
                    value=[],
                    multi=True)
                ], 
                style={'marginBottom': '6%'}
            ), 

            html.Div(children=[
                html.H6('Select Type(s):', className='card-text'),
                dcc.Dropdown(
                    id='specific_type_dropdown', 
                    options=graph.type_dropdown_options_initial,
                    value=[],
                    multi=True)
                ], 
                style={'marginBottom': '6%'}
            )          
        ]
    )
)

# DBC card that defines graph manipulation layout.
graph_manipulation = dbc.Card(
    dbc.CardBody(
        [   
            html.H6('Target Spread:', className='card-text', style={'marginBottom': '3%'}),
            html.Div(
                dcc.Slider(
                    id='target_spread_slider',
                    min=0.1,
                    max=3,
                    step=0.1,
                    value=graph.target_spread,
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ),
                style={'marginBottom': '6%'}
            ),

            html.H6('Source Spread:', className='card-text', style={'marginBottom': '3%'}),
            html.Div(
                dcc.Slider(
                    id='source_spread_slider',
                    min=0.01,
                    max=0.3,
                    step=0.01,
                    value=graph.source_spread,
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ),
                style={'marginBottom': '6%'}
            ),

            html.H6('Node Size:', className='card-text', style={'marginBottom': '3%'}),
            html.Div(
                dcc.Slider(
                    id='node_size_slider',
                    min=0,
                    max=1,
                    step=0.01,
                    value=graph.node_size_modifier,
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ),
                style={'marginBottom': '6%'}
            ),

            html.H6('Edge Size:', className='card-text', style={'marginBottom': '3%'}),
            html.Div(
                dcc.Slider(
                    id='edge_size_slider',
                    min=0,
                    max=1,
                    step=0.01,
                    value=graph.edge_size_modifier,
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ),
                style={'marginBottom': '6%'}
            ),

            html.H6('Simulation Iterations:', className='card-text', style={'marginBottom': '3%'}),
            html.Div(
                dcc.Slider(
                    id='simulation_iterations_slider',
                    min=0,
                    max=50,
                    step=1,
                    value=graph.simulation_iterations,
                    tooltip={'placement': 'bottom', 'always_visible': True},
                    vertical=False
                ),
                style={'marginBottom': '6%'}
            ),

            html.Div(children=[
                dbc.Button(
                    'Simulate', 
                    id='simulate_button', 
                    style={'background': 'linear-gradient(#8a5a00, #b17400 40%, #8a5a00)'}, 
                )
                ],
                style={'marginBottom': '4%'}, 
                className='d-grid gap-2 col-6 mx-auto'
            )
        ]
    )
)

# DBC card that defines color editing layout.
color_editing = dbc.Card(
    dbc.CardBody(
        [
            html.Div(children=[
                html.H6('Type Gradient:', style={'marginRight': '5%'}, className='card-text'
                ),
                dbc.Input(
                    type='color',
                    id='gradient_start',
                    value=graph.gradient_start_initial,
                    style={'width': 50, 'height': 25, 'display': 'inline-block', 'marginRight': '1%', 'border': 'none', 'padding': '0'}
                ),
                html.Div(',', style={'display': 'inline-block', 'marginRight': '1%', 'marginLeft': '1%'}),
                dbc.Input(
                    type='color',
                    id='gradient_end',
                    value=graph.gradient_end_initial,
                    style={'width': 50, 'height': 25, 'display': 'inline-block', 'marginLeft': '1%', 'border': 'none', 'padding': '0'},
                )               
                ], 
                style={'display': 'flex', 'justiftyContent': 'center', 'marginBottom': '6%'}
            ), 

            html.Div(children=[
                html.H6('Selected Source Node Type Color:', style={'marginRight': '5%'}, className='card-text'),
                dbc.Input(
                    type='color',
                    id='selected_type_color',
                    value=graph.selected_type_color_initial,
                    style={'width': 50, 'height': 25, 'display': 'inline-block', 'border': 'none', 'padding': '0'}
                )           
                ], 
                style={'display': 'flex', 'justiftyContent': 'center', 'marginBottom': '6%'}
            ), 

            html.Div(children=[
                html.H6('Source Node Color:', style={'marginRight': '5%'}, className='card-text'),
                dbc.Input(
                    type='color',
                    id='source_color',
                    value=graph.target_color_initial,
                    style={'width': 50, 'height': 25, 'display': 'inline-block', 'border': 'none', 'padding': '0'}
                )           
                ], 
                style={'display': 'flex', 'justiftyContent': 'center', 'marginBottom': '6%'}
            ), 

            html.Div(children=[
                html.H6('Target Node Color:', style={'marginRight': '5%'}, className='card-text'),
                dbc.Input(
                    type='color',
                    id='target_color',
                    value=graph.target_color_initial,
                    style={'width': 50, 'height': 25, 'display': 'inline-block', 'border': 'none', 'padding': '0'}
                )           
                ], 
                style={'display': 'flex', 'justiftyContent': 'center', 'marginBottom': '6%'}
            ),

            html.Div(children=[
                dbc.Button(
                    'Randomize Colors', 
                    id='randomize_colors_button', 
                    style={'background': 'linear-gradient(#8a5a00, #b17400 40%, #8a5a00)'}
                )
                ],
                style={'marginBottom': '4%'}, 
                className='d-grid gap-2 col-6 mx-auto'
            )
        ]
    ),
)

# DBC card that defines individual node selection layout.
node_data = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                html.Div(
                    children='Select Node(s)',
                    id='node_data'
                ), 
                style={'max-height': '400px', 'overflowX': 'scroll', 'overflowY': 'scroll'}
            )
        ]
    )
)

# DBC card that defines table layout.
table_data = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                dt.DataTable(
                    id='data_table', 
                    columns=graph.data_table_columns, 
                    data=graph.table_data, 
                    style_as_list_view=True,
                    style_cell={
                        'backgroundColor': '#32383E', 
                        'textAlign': 'left'
                    },
                    style_header={
                        'border': '#32383E'
                    },
                    style_table={'overflowX': 'auto', 'overflowY': 'auto', 'max-height': '400px'},
                    style_data_conditional=[                
                        {
                            'if': {'state': 'selected'},
                            'backgroundColor': '#8ecae6',
                            'border': '#FFFFFF',
                            'color': '#000000'
                        }
                    ],
                    css=[
                        { 'selector': '.current-page', 'rule': 'visibility: hidden;'}, 
                        { 'selector': '.current-page-shadow', 'rule': 'color: #AAAAAA; font-size: 16px;'}, 
                        { 'selector': '.next-page:hover', 'rule': 'color: #8ecae6;'}, 
                        { 'selector': '.last-page:hover', 'rule': 'color: #8ecae6;'}, 
                        { 'selector': '.previous-page:hover', 'rule': 'color: #8ecae6;'}, 
                        { 'selector': '.first-page:hover', 'rule': 'color: #8ecae6;'}
                    ],
                    export_format='csv',
                    page_size=50
                )
            )
        ]
    )
)

# Defines Cytoscape stylesheet.
default_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'width': 'data(size)',
            'height': 'data(size)',
            'content': 'data(label)',
            'font-size': 'data(label_size)',
            'color': '#dad9d6', 
            'background-color': 'data(color)'
            }
        },
    {
        'selector': 'edge',
        'style': {
            'width': 'data(size)', 
            'line-opacity': 'data(size)', 
            'label': 'data(label)', 
            'font-size': 'data(label_size)', 
            'color': '#dad9d6', 
            'font-opacity': 1
            }
        }, 
    {
        'selector': ':selected',
        'style': {
            'border-color': 'black', 
            'border-opacity': '1', 
            'border-width': '0.075px'
            }
        }
    ]

# Cytoscape graph object.
cytoscape_graph = cyto.Cytoscape(
    id='output_graph',
    layout={'name': 'preset'},
    style={'width': '100vw', 'height': '100vh'},
    stylesheet=default_stylesheet,
    elements=graph.elements,
    boxSelectionEnabled=True
    )

def format_data_input(csv_input, filename):
    """
    Function that converts encoded data upload into graph-usable dataframe.

    Args:
        csv_input (string): Base64 encoded data.
        filename (string): Uploaded data filename.

    """

    for _, data in enumerate(csv_input):
        _, content_string = data.split(',')

        decoded = base64.b64decode(content_string)

        try:
            decoded_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), encoding='utf-8', index_col=0)

        except Exception as e:
            print(e)

    return decoded_df

def server_layout():
    """
    Function that houses the app layout. Calling the function allows the UI state and graph to be reset.

    """
    ui_tracker.set_initial_state()

    app_layout = html.Div([
        html.Div(cytoscape_graph, style={'position': 'fixed', 'zIndex': '1', 'width': '99vw', 'height': '99vh'}),
        
        html.Div(children=[
            html.Div(children=[
                dbc.Button(children=[
                    'Expand Settings'
                    ],
                    id='settings_button',
                    className=ui_tracker.settings_button_class,
                    style={'width': '50%'}
                ), 
                dbc.Button(children=[
                    'Reset Graph'
                    ],
                    id='reset_button',
                    style={'width': '50%', 'background': 'linear-gradient(#8a0000, #af0000 40%, #8a0000'}
                ) 
                ], 
                style={'width': '20vw', 'background': '#272B30'}
            ),
            
            dbc.Collapse(children=[
                dbc.Button(
                    'Graph Sliders',
                    id='graph_sliders_button',
                    className=ui_tracker.graph_sliders_button_class,
                    style={'width': '20vw'}
                ), 
                dbc.Collapse(
                    graph_sliders,
                    id='graph_sliders_collapse',
                    is_open=False
                ), 
                dbc.Button(
                    'Node Filtering',
                    id='node_filtering_button',
                    className=ui_tracker.node_filtering_button_class,
                    style={'width': '20vw'}
                ), 
                dbc.Collapse(
                    node_filtering,
                    id='node_filtering_collapse',
                    is_open=False
                ), 
                dbc.Button(
                    'Graph Manipulation',
                    id='graph_manipulation_button',
                    className=ui_tracker.graph_manipulation_button_class,
                    style={'width': '20vw'}
                ), 
                dbc.Collapse(
                    graph_manipulation,
                    id='graph_manipulation_collapse',
                    is_open=False
                ), 
                dbc.Button(
                    'Color Editing',
                    id='color_editing_button',
                    className=ui_tracker.color_editing_button_class,
                    style={'width': '20vw'}
                ), 
                dbc.Collapse(
                    color_editing,
                    id='color_editing_collapse',
                    is_open=False
                ), 
                dbc.Button(
                    'Node Data',
                    id='node_data_button',
                    className=ui_tracker.node_data_button_class,
                    style={'width': '20vw'}
                ), 
                dbc.Collapse(
                    node_data,
                    id='node_data_collapse',
                    is_open=False
                ), 
                dbc.Button(
                    'Table Data',
                    id='table_data_button',
                    className=ui_tracker.table_data_button_class,
                    style={'width': '20vw'}
                ), 
                dbc.Collapse(
                    table_data,
                    id='table_data_collapse',
                    is_open=False
                ), 

                html.Div(children=[
                    dcc.Upload(
                        dbc.Button(
                            'Upload Data', 
                            style={'width': '10vw', 'background': 'linear-gradient(#008a07, #00af0e 40%, #008a07)'}
                        ),
                        id='data_upload', 
                        multiple=True
                    ), 
                    dbc.Button(
                        'Download Image', 
                        style={'width': '10vw', 'background': 'linear-gradient(#4a018a, #5f00b3 40%, #4a018a)'}, 
                        id='download_image', 
                    )
                    ], 
                    style={'display': 'flex'}
                )
                ],
                id='settings_collapse',
                is_open=False, 
                style={'width': '20vw'}
            )
            
            ], 
            style={
                'display': 'flex', 'flex-direction': 'column', 'marginLeft': '1vw', 
                'marginTop': '1vh', 'width': 'fit-content', 'padding': '0px', 'background': '#32383E', 'border': 'none', 
                'position': 'relative', 'zIndex': '22'}
        )
        
    ])
    
    return app_layout

app.layout = server_layout

# Button callback
@app.callback(
    Output('settings_collapse', 'is_open'),
    Output('settings_button', 'className'),
    Output('settings_button', 'children'),
    Input('settings_button', 'n_clicks')
)
def toggle_settings(settings_button_clicks):
    if settings_button_clicks:
        ui_tracker.settings_button_toggle = not ui_tracker.settings_button_toggle

        if ui_tracker.settings_button_toggle:
            ui_tracker.settings_button_class = 'button_enabled'
            ui_tracker.settings_button_text = 'Collapse Settings'

        else:
            ui_tracker.settings_button_class = 'button_disabled'
            ui_tracker.settings_button_text = 'Expand Settings'
    
    return [ui_tracker.settings_button_toggle, ui_tracker.settings_button_class, ui_tracker.settings_button_text]

# Button callback
@app.callback(
    Output('graph_sliders_collapse', 'is_open'),
    Output('graph_sliders_button', 'className'),
    Input('graph_sliders_button', 'n_clicks')
)
def toggle_sliders(graph_sliders_button_clicks):
    if graph_sliders_button_clicks:
        ui_tracker.graph_sliders_button_toggle = not ui_tracker.graph_sliders_button_toggle

        if ui_tracker.graph_sliders_button_toggle:
            ui_tracker.graph_sliders_button_class = 'button_enabled'

        else:
            ui_tracker.graph_sliders_button_class = 'button_disabled'
    
    return [ui_tracker.graph_sliders_button_toggle, ui_tracker.graph_sliders_button_class]

# Button callback
@app.callback(
    Output('node_filtering_collapse', 'is_open'),
    Output('node_filtering_button', 'className'),
    Input('node_filtering_button', 'n_clicks')
)
def toggle_filtering(node_filtering_button_clicks):
    if node_filtering_button_clicks:
        ui_tracker.node_filtering_button_toggle = not ui_tracker.node_filtering_button_toggle

        if ui_tracker.node_filtering_button_toggle:
            ui_tracker.node_filtering_button_class = 'button_enabled'

        else:
            ui_tracker.node_filtering_button_class = 'button_disabled'

    return [ui_tracker.node_filtering_button_toggle, ui_tracker.node_filtering_button_class]

# Button callback
@app.callback(
    Output('graph_manipulation_collapse', 'is_open'),
    Output('graph_manipulation_button', 'className'),
    Input('graph_manipulation_button', 'n_clicks')
)
def toggle_manipulation(graph_manipulation_button_clicks):
    if graph_manipulation_button_clicks:
        ui_tracker.graph_manipulation_button_toggle = not ui_tracker.graph_manipulation_button_toggle

        if ui_tracker.graph_manipulation_button_toggle:
            ui_tracker.graph_manipulation_button_class = 'button_enabled'

        else:
            ui_tracker.graph_manipulation_button_class = 'button_disabled'

    return [ui_tracker.graph_manipulation_button_toggle, ui_tracker.graph_manipulation_button_class]

# Button callback
@app.callback(
    Output('color_editing_collapse', 'is_open'),
    Output('color_editing_button', 'className'),
    Input('color_editing_button', 'n_clicks')
)
def toggle_coloring(color_editing_button_clicks):
    if color_editing_button_clicks:
        ui_tracker.color_editing_button_toggle = not ui_tracker.color_editing_button_toggle

        if ui_tracker.color_editing_button_toggle:
            ui_tracker.color_editing_button_class = 'button_enabled'

        else:
            ui_tracker.color_editing_button_class = 'button_disabled'

    return [ui_tracker.color_editing_button_toggle, ui_tracker.color_editing_button_class]

# Button callback
@app.callback(
    Output('node_data_collapse', 'is_open'),
    Output('node_data_button', 'className'),
    Input('node_data_button', 'n_clicks')
)
def toggle_node_data(node_data_button_clicks):
    if node_data_button_clicks:
        ui_tracker.node_data_button_toggle = not ui_tracker.node_data_button_toggle

        if ui_tracker.node_data_button_toggle:
            ui_tracker.node_data_button_class = 'button_enabled'

        else:
            ui_tracker.node_data_button_class = 'button_disabled'

    return [ui_tracker.node_data_button_toggle, ui_tracker.node_data_button_class]

# Button callback
@app.callback(
    Output('table_data_collapse', 'is_open'),
    Output('table_data_button', 'className'),
    Input('table_data_button', 'n_clicks')
)
def toggle_table_data(table_data_button_clicks):
    if table_data_button_clicks:
        ui_tracker.table_data_button_toggle = not ui_tracker.table_data_button_toggle

        if ui_tracker.table_data_button_toggle:
            ui_tracker.table_data_button_class = 'button_enabled'

        else:
            ui_tracker.table_data_button_class = 'button_disabled'

    return [ui_tracker.table_data_button_toggle, ui_tracker.table_data_button_class]

# Main callback that enables graph updates.
@app.callback(
    Output(component_id='output_graph', component_property='elements'),
    Output(component_id='combined_value_range_slider', component_property='min'),
    Output(component_id='combined_value_range_slider', component_property='max'),
    Output(component_id='combined_value_range_slider', component_property='step'), 
    Output(component_id='combined_value_range_slider', component_property='value'),
    Output(component_id='edge_value_range_slider', component_property='min'),
    Output(component_id='edge_value_range_slider', component_property='max'),
    Output(component_id='edge_value_range_slider', component_property='step'),
    Output(component_id='edge_value_range_slider', component_property='value'),
    Output(component_id='max_node_slider', component_property='max'), 
    Output(component_id='max_node_slider', component_property='value'), 
    Output(component_id='specific_target_dropdown', component_property='options'),
    Output(component_id='specific_target_dropdown', component_property='value'),
    Output(component_id='specific_source_dropdown', component_property='options'), 
    Output(component_id='specific_source_dropdown', component_property='value'),
    Output(component_id='specific_type_dropdown', component_property='options'), 
    Output(component_id='specific_type_dropdown', component_property='value'), 
    Output(component_id='target_spread_slider', component_property='value'),
    Output(component_id='source_spread_slider', component_property='value'),
    Output(component_id='node_size_slider', component_property='value'),
    Output(component_id='edge_size_slider', component_property='value'), 
    Output(component_id='simulation_iterations_slider', component_property='value'),
    Output(component_id='gradient_start', component_property='value'), 
    Output(component_id='gradient_end', component_property='value'), 
    Output(component_id='selected_type_color', component_property='value'), 
    Output(component_id='source_color', component_property='value'),
    Output(component_id='target_color', component_property='value'),
    Output(component_id='data_table', component_property='columns'),
    Output(component_id='data_table', component_property='data'),
    Output(component_id='data_upload', component_property='contents'),
    Output(component_id='data_upload', component_property='filename'),
    Output(component_id='output_graph', component_property='generateImage'),
    [Input(component_id='combined_value_range_slider', component_property='value')],
    [Input(component_id='edge_value_range_slider', component_property='value')],
    Input(component_id='max_node_slider', component_property='value'),
    Input(component_id='specific_target_dropdown', component_property='value'), 
    Input(component_id='specific_source_dropdown', component_property='value'), 
    Input(component_id='specific_type_dropdown', component_property='value'), 
    Input(component_id='target_spread_slider', component_property='value'), 
    Input(component_id='source_spread_slider', component_property='value'), 
    Input(component_id='node_size_slider', component_property='value'),
    Input(component_id='edge_size_slider', component_property='value'), 
    State(component_id='simulation_iterations_slider', component_property='value'),
    Input(component_id='simulate_button', component_property='n_clicks'), 
    Input(component_id='gradient_start', component_property='value'), 
    Input(component_id='gradient_end', component_property='value'), 
    Input(component_id='selected_type_color', component_property='value'), 
    Input(component_id='source_color', component_property='value'),
    Input(component_id='target_color', component_property='value'), 
    Input(component_id='randomize_colors_button', component_property='n_clicks'), 
    Input(component_id='reset_button', component_property='n_clicks'), 
    Input(component_id='data_upload', component_property='contents'),
    Input(component_id='data_upload', component_property='filename'), 
    Input(component_id='download_image', component_property='n_clicks'),
    prevent_initial_call=True
)
def toggle_graph_update(
    input_combined_value_range_slider, 
    input_edge_value_range_slider, 
    input_max_node_slider, 
    input_specific_target_dropdown, 
    input_specific_source_dropdown, 
    input_specific_type_dropdown, 
    input_target_spread,
    input_sn_spread, 
    input_node_size, 
    input_edge_size, 
    input_simulation_iterations, 
    input_simulate_button, 
    input_gradient_start, 
    input_gradient_end, 
    input_selected_type_color, 
    input_source_color, 
    input_target_color, 
    input_randomize_colors_button_clicks, 
    input_reset_button, 
    data_upload_content, 
    data_upload_filenames, 
    download_graph_clicks_input):

    if test_timing:
        start_time = time.time()
        print('================')

    if input_combined_value_range_slider != graph.combined_value_range:
        graph.combined_value_range = input_combined_value_range_slider

        graph.graph_update_shortcut = False

    if input_edge_value_range_slider != graph.edge_value_range:
        graph.edge_value_range = input_edge_value_range_slider

        graph.graph_update_shortcut = False

    if input_max_node_slider != graph.max_node_count:
        graph.max_node_count = input_max_node_slider

        graph.graph_update_shortcut = False

    if input_specific_target_dropdown != graph.target_dropdown_selection:
        graph.target_dropdown_selection = input_specific_target_dropdown

        graph.graph_update_shortcut = False

    if input_specific_source_dropdown != graph.source_dropdown_selection:
        graph.source_dropdown_selection = input_specific_source_dropdown

        graph.graph_update_shortcut = False

    if input_specific_type_dropdown != graph.type_dropdown_selection:
        graph.type_dropdown_selection = input_specific_type_dropdown

        graph.graph_update_shortcut = False

    if input_target_spread != graph.target_spread:
        graph.target_spread = input_target_spread

    if input_sn_spread != graph.source_spread:
        graph.source_spread = input_sn_spread

    if input_node_size != graph.node_size_modifier:
        graph.node_size_modifier = input_node_size

    if input_edge_size != graph.edge_size_modifier:
        graph.edge_size_modifier = input_edge_size

    if input_simulation_iterations != graph.simulation_iterations:
        graph.simulation_iterations = input_simulation_iterations

    graph.target_spread = input_target_spread
    graph.source_spread = input_sn_spread

    if (input_gradient_start != graph.gradient_start) and (input_gradient_start != graph.gradient_start_initial):
        ui_tracker.display_gradient_start_color = input_gradient_start

        if input_gradient_end != graph.gradient_end_initial:
            graph.gradient_start = input_gradient_start
            graph.gradient_end = input_gradient_end
            graph.gradient_color_primacy = True

            ui_tracker.display_gradient_end_color = input_gradient_end
        
    if (input_gradient_end != graph.gradient_end) and (input_gradient_end != graph.gradient_end_initial):
        ui_tracker.display_gradient_end_color = input_gradient_end

        if input_gradient_start != graph.gradient_start_initial:
            graph.gradient_start = input_gradient_start
            graph.gradient_end = input_gradient_end
            graph.gradient_color_primacy = True

            ui_tracker.display_gradient_start_color = input_gradient_start

    if (input_selected_type_color != graph.selected_type_color) and (input_selected_type_color != graph.selected_type_color_initial):
        graph.selected_type_color = input_selected_type_color

        ui_tracker.display_selected_type_color = input_selected_type_color
        graph.type_color_primacy = True

    if (input_source_color != graph.source_color) and (input_source_color != graph.source_color_initial):
        graph.source_color = input_source_color

        ui_tracker.display_source_color = input_source_color
        graph.source_color_primacy = True

    if (input_target_color != graph.target_color) and (input_target_color != graph.target_color_initial):
        graph.target_color = input_target_color

        ui_tracker.display_target_color = input_target_color
        graph.target_color_primacy = True

    if input_randomize_colors_button_clicks != ui_tracker.randomized_color_button_clicks:
        ui_tracker.randomized_color_button_clicks = input_randomize_colors_button_clicks

        ui_tracker.display_gradient_start_color = graph.gradient_start_initial
        ui_tracker.display_gradient_end_color = graph.gradient_end_initial
        ui_tracker.display_selected_type_color = graph.selected_type_color_initial
        ui_tracker.display_source_color = graph.source_color_initial
        ui_tracker.display_target_color = graph.target_color_initial
        
        graph.random_color_primacy = True

    if download_graph_clicks_input != ui_tracker.download_graph_clicks:
        ui_tracker.download_graph_clicks = download_graph_clicks_input

        image_download = {'type': 'jpg', 'zoom': '200%', 'action': 'download'}

    else:
        image_download = {'action': ''}

    if data_upload_content != None:
        csv_user_input = format_data_input(data_upload_content, data_upload_filenames)
        elements = graph.load_additional_data(csv_user_input)

        ui_tracker.display_gradient_start_color = graph.gradient_start_initial
        ui_tracker.display_gradient_end_color = graph.gradient_end_initial
        ui_tracker.display_selected_type_color = graph.selected_type_color_initial
        ui_tracker.display_source_color = graph.source_color_initial
        ui_tracker.display_target_color = graph.target_color_initial

    elif input_reset_button != ui_tracker.reset_button_clicks:
        ui_tracker.reset_button_clicks = input_reset_button

        graph.reset_graph()

        elements = graph.starting_elements

        ui_tracker.display_gradient_start_color = graph.gradient_start_initial
        ui_tracker.display_gradient_end_color = graph.gradient_end_initial
        ui_tracker.display_selected_type_color = graph.selected_type_color_initial
        ui_tracker.display_source_color = graph.source_color_initial
        ui_tracker.display_target_color = graph.target_color_initial

    elif input_simulate_button != ui_tracker.simulate_button_clicks:
        ui_tracker.simulate_button_clicks = input_simulate_button
        graph.simulate()
        elements = graph.elements

    else:
        elements = graph.update_graph_elements()

    if test_timing:
        print('TOTAL TIME UPDATE: ' + str(time.time() - start_time))
        print('================')

    return [
        elements,
        graph.combined_value_bound[0],
        graph.combined_value_bound[1],
        graph.combined_value_step_size, 
        graph.combined_value_range, 
        graph.edge_value_bound[0],
        graph.edge_value_bound[1],
        graph.edge_value_step_size,
        graph.edge_value_range, 
        graph.max_node_count_initial, 
        graph.max_node_count, 
        graph.target_dropdown_options,
        graph.target_dropdown_selection, 
        graph.source_dropdown_options, 
        graph.source_dropdown_selection, 
        graph.type_dropdown_options, 
        graph.type_dropdown_selection, 
        graph.target_spread, 
        graph.source_spread,
        graph.node_size_modifier,
        graph.edge_size_modifier, 
        graph.simulation_iterations, 
        ui_tracker.display_gradient_start_color, 
        ui_tracker.display_gradient_end_color, 
        ui_tracker.display_selected_type_color,
        ui_tracker.display_source_color, 
        ui_tracker.display_target_color, 
        graph.data_table_columns, 
        graph.table_data, 
        None, 
        None, 
        image_download
        ]

# Individual node data callback, dissociated with main update callback due to lack of actual graph change.
@app.callback(
    Output('node_data', 'children'),
    Input('output_graph', 'selectedNodeData'))
def displayTapNodeData(input_selected_nodes):

    if (input_selected_nodes == []) or (input_selected_nodes == None):
        return 'Select Node(s)'

    formatted_node_data = graph.generate_node_data(input_selected_nodes)

    display_data = []

    for i, node in enumerate(input_selected_nodes):
        display_data.append(html.Div(
            children=[
                node['label']
                ],
                style={'font-weight': 'bold'}
            )
        )

        display_data.append(html.Pre(formatted_node_data[i]))

    return display_data

if test_timing:
    print('TOTAL TIME STARTUP: ' + str(time.time() - start_time_global))
    print('================')

if __name__ == '__main__':
    app.run_server(debug=True)