import os
from format_semnet_data import composite_view_data_from_semnet

# Won't work if formatted_cv_data.csv (or any other CSV) already exists in current directory
composite_view_data_from_semnet(os.getcwd() + '\semnet_results')