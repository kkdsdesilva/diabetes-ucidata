# script to preprocess data

import sys
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the root directory
root_dir = os.path.join(current_dir, '../')

# Add the root directory to the Python path
sys.path.append(root_dir)

from src.data.data_collection import import_data
from src.data.data_preprocessing import add_dtypes, data_wo_weight, data_w_weight, impute_data
from src.data.save_data import save_data

def main():
    """Preprocesses the data and saves it to a CSV file."""
    
    # Import data
    data = import_data()

    # Preprocess data
    data = add_dtypes(data)
    data_without_weight = data_wo_weight(data)
    data_with_weight = data_w_weight(data)

    #impute data
    data_without_weight_imputed = impute_data(data_without_weight)
    data_with_weight_imputed = impute_data(data_with_weight)

    # save data
    save_data(data_without_weight_imputed, root_dir+'/data/preprocessed', 'diabetes_without_weight_cleaned.csv')
    save_data(data_with_weight_imputed, root_dir+'/data/preprocessed', 'diabetes_with_weight_cleaned.csv')


if __name__ == '__main__':
    main()