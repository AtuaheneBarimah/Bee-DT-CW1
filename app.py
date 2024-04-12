# Import Python Lib. You can add if required
import numpy as np
import pandas as pd

# Connect Google Colab to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create python class to read datasets
columns_to_drop = ['Column_1',... 'Column_N']

class KeyDataFrame(pd.DataFrame):
    def __repr__(self):
        return super().__repr__()

def process_lvm_file(file_path, week_name):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    variable_name = '_'.join(file_name.split('_')[:2])
    df = pd.read_csv(file_path, delimiter='\t', skiprows=23, header=None)
    column_names = ['Column_1',... 'Column_N'] #Columns to keep
    df.columns = column_names
    df = df.drop(columns=columns_to_drop)
    data_frames[week_name][variable_name] = KeyDataFrame(df)

class DataFrameDict(dict):
    def __missing__(self, key):
        self[key] = {}
        return self[key]
data_frames = DataFrameDict()
week_folders = ['/content/drive/MyDrive/FC0',
                '/content/drive/MyDrive/FC1',
                '/content/drive/MyDrive/FC2',
                '/content/drive/MyDrive/FC3',
                '/content/drive/MyDrive/FC4',
                '/content/drive/MyDrive/FC5']

for week_folder in week_folders:
    week_name = Path(week_folder).name
    for root, dirs, files in os.walk(week_folder):
        for file in files:
            if file.endswith(".lvm"):
                file_path = os.path.join(root, file)
                process_lvm_file(file_path, week_name)

globals().update(data_frames)


# Create a function to rename all the columns and adjust pressure values
def replace_and_modify_columns(df, x):
    # Replace column names 
    df.rename(columns={'Bee_1': 'Cee_1',
                       'Bee_2': 'Cee_2',
                       'Bee_3': 'Cee_3',
                       'Bee_4': 'Cee_4',
                       'Bee_5': 'Cee_5'}, inplace=True)
    # Modify values in Cee_2 and Cee_5
    df['Cee_2'] += x
    df['Cee_5'] += x
    return df

FC1_1 =replace_and_modify_columns(((FC1['File_name'])),1.0101) # Example use assuming x = 1.0101

#
