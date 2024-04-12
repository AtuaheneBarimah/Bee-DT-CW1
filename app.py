# Import Python Lib. You can add if required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

FC1_1 =replace_and_modify_columns(((FC1['File_name'])),1.0101) # Example use assuming x = 1.0101 and FC1 is your folder name. 

# Plot Histogram for various parameters
S1 = (FC0_1['File_name'])
S2 = (FC1_1['File_name'])
S3 = (FC1_2['File_name'])
S4 = (FC1_3['File_name'])
num_bins = 20
plt.figure(figsize=(10, 6))
sns.histplot(S1.P1_bar, kde=True, bins=num_bins, label='P1(bar)@Week 1', color='blue', alpha=0.7)
sns.histplot(S2.P1_bar, kde=True, bins=num_bins, label='P1(bar)@Week 2', color='green', alpha=0.7)
sns.histplot(S3.P1_bar, kde=True, bins=num_bins, label='P1(bar)@Week 3', color='red', alpha=0.7)
sns.histplot(S4.P1_bar, kde=True, bins=num_bins, label='P1(bar)@Week 4', color='purple', alpha=0.7)
plt.xlabel('Values (bar)')
plt.ylabel('Distribution')
plt.title('Distribution of P1(bar)')
plt.legend()
plt.grid(True)
plt.show()


# Create Spider Diagrams

# Means for Healthy condition. Just remem to concartinate all the data in the HC0 folder
HC0_1_mean = HC0_1.mean()

# Dummy placeholder values for means of P1,...P5
FC1 = {'FC1_1': FC1_1, 'FC1_2': FC1_2, 'FC1_3': FC1_3, 'FC1_4': FC1_4, 'FC1_5': FC1_5}
FC2 = {'FC2_1': FC2_1, 'FC2_2': FC2_2, 'FC2_3': FC2_3, 'FC2_4': FC2_4, 'FC2_5': FC2_5}
FC3 = {'FC3_1': FC3_1, 'FC3_2': FC3_2, 'FC3_3': FC3_3, 'FC3_4': FC3_4, 'FC3_5': FC3_5}
FC4 = {'FC4_1': FC4_1, 'FC4_2': FC4_2, 'FC4_3': FC4_3, 'FC4_4': FC4_4, 'FC4_5': FC4_5}
FC5 = {'FC5_1': FC5_1, 'FC5_2': FC5_2, 'FC5_3': FC5_3, 'FC5_4': FC5_4, 'FC5_5': FC5_5}

# Initialize dictionaries to store the means
FC1_means = {}
FC2_means = {}
FC3_means = {}
FC4_means = {}
FC5_means = {}

# Calculate means for each category
for category_dict in [FC1, FC2, FC3, FC4, FC5]:
    category_means = {}
    for key, value in category_dict.items():
        category_means[key] = value.mean()
    if category_dict is FC1:
        FC1_means = category_means
    elif category_dict is FC2:
        FC2_means = category_means
    elif category_dict is FC3:
        FC3_means = category_means
    elif category_dict is FC4:
        FC4_means = category_means
    elif category_dict is FC5:
        FC5_means = category_means

