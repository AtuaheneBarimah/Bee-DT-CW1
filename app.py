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

FC1_1 =replace_and_modify_columns(((FC1['File_name'])),1.0101) # Example use assuming x = 1.0101 and FC1 is your folder name

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
FC1_1_mean = FC1_1.mean()
FC1_2_mean = FC1_2.mean()
FC1_3_mean = FC1_3.mean()
FC1_4_mean = FC1_4.mean()
FC1_5_mean = FC1_5.mean()

FC2_1_mean = FC1_1.mean()
FC2_2_mean = FC1_2.mean()
FC2_3_mean = FC1_3.mean()
FC2_4_mean = FC1_4.mean()
FC2_5_mean = FC1_5.mean()

FC3_1_mean = FC1_1.mean()
FC3_2_mean = FC1_2.mean()
FC3_3_mean = FC1_3.mean()
FC3_4_mean = FC1_4.mean()
FC3_5_mean = FC1_5.mean()

FC4_1_mean = FC1_1.mean()
FC4_2_mean = FC1_2.mean()
FC4_3_mean = FC1_3.mean()
FC4_4_mean = FC1_4.mean()
FC4_5_mean = FC1_5.mean()

FC5_1_mean = FC1_1.mean()
FC5_2_mean = FC1_2.mean()
FC5_3_mean = FC1_3.mean()
FC5_4_mean = FC1_4.mean()
FC5_5_mean = FC1_5.mean()

def spider_check_deviation (x0, x1,x2,x3,x4,x5):
    A_1 = (abs(x0-x1))/x0

    return 

# Creating the DataFrame
Algo_data = {
    'Component': ['Filter', 'Pump', 'Valve', 'Nozzle', 'Pipe'],
    'SPC_Model': [results_df.SPC_Model_Performance[0], results_df.SPC_Model_Performance[1], results_df.SPC_Model_Performance[2], results_df.SPC_Model_Performance[3], results_df.SPC_Model_Performance[4]],
    'Ensemble_Classifiers': [best_accuracy_1, best_accuracy_2, best_accuracy_3, best_accuracy_4, best_accuracy_5],
    'Neural_Network_Model': [DL_best_accuracy_1, DL_best_accuracy_2, DL_best_accuracy_3, DL_best_accuracy_4, DL_best_accuracy_5],
    'Approximation_Model': [AENS_HDT_Per_1, AENS_HDT_Per_2, AENS_HDT_Per_3, AENS_HDT_Per_4, AENS_HDT_Per_5],
    'PINN_Model': [HDT_PINN_1, HDT_PINN_2, HDT_PINN_3, HDT_PINN_4, HDT_PINN_5],
    'Ensemble_HDT_Model': [ENS_HDT_Per_1, ENS_HDT_Per_2, ENS_HDT_Per_3, ENS_HDT_Per_4, ENS_HDT_Per_5]
}

Algo_data_df = pd.DataFrame(Algo_data)
print(Algo_data_df)

