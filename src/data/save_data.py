root_path = '../../'

import sys
sys.path.append(root_path)
from src.data.data_collection import import_data

# import data
data = import_data()

# save data
data.to_csv(root_path+'data/raw/diabetes.csv', index=False)