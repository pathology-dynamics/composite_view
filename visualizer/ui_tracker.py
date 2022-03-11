class UI_Tracker:
    '''
    Class for tracking all UI elements that change throughout app run.

    '''

    def __init__(self, json_data=False):

        if not json_data:
            self.set_initial_state()

        else:
            self.load_attributes(json_data)

    def set_initial_state(self):
        '''
        Initializes all UI states that don't directly interact with the graph object.

        '''

        self.card_stack_tracking = []
        self.dropdown_cards = []

        self.settings_button_clicks = None
        self.settings_button_toggle = False
        self.settings_button_class = 'button_disabled'
        self.settings_button_text = 'Expand Settings'

        self.graph_sliders_button_clicks = None
        self.graph_sliders_button_toggle = False
        self.graph_sliders_button_class = 'button_disabled'

        self.node_filtering_button_clicks = None
        self.node_filtering_button_toggle = False
        self.node_filtering_button_class = 'button_disabled'

        self.graph_manipulation_button_clicks = None
        self.graph_manipulation_button_toggle = False
        self.graph_manipulation_button_class = 'button_disabled'

        self.color_editing_button_clicks = None
        self.color_editing_button_toggle = False
        self.color_editing_button_class = 'button_disabled'

        self.node_data_button_clicks = None
        self.node_data_button_toggle = False
        self.node_data_button_class = 'button_disabled'

        self.table_data_button_clicks = None
        self.table_data_button_toggle = False
        self.table_data_button_class = 'button_disabled'

        self.reset_button_clicks = None
        self.simulate_button_clicks = None
        self.randomized_color_button_clicks = None

        self.color_change_only = True
        self.graph_manipulation_only = True

        self.download_graph_clicks = None

        self.display_gradient_start_color = ''
        self.display_gradient_end_color = ''
        self.display_selected_type_color = ''
        self.display_source_color = ''
        self.display_target_color = ''

    def load_attributes(self, json_data):
        
        self.__dict__ = dict(json_data)