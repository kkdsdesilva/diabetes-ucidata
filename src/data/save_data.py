# Function: save_data

def save_data(data, file_path, filename):

    """Saves data to a CSV file.
    
    Parameters:
    data (DataFrame): The data to save.
    file_path (str): The path of the file.
    filename (str): The name of the CSV file to save the data to.
    """
    
    # Save data
    data.to_csv(file_path+'/'+filename, index=False)