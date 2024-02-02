root_path = '../../'

import sys
sys.path.append(root_path)

from src.data.data_collection import import_data

# import data
data = import_data()

print(data.head())