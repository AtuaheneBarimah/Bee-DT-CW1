# Import Python Lib. You can add if required
!pip install --upgrade keras scikit-learn
!pip install cairosvg
import cairosvg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, concatenate
from keras.metrics import mean_absolute_error, mean_absolute_percentage_error


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


# Create a function to rename all the columns and adjust pressure values. See example below
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
Algo_data_df

radar_chart =pygal.Radar(width = 350, height = 250)
radar_chart.title = 'Compare Performance of FDI Model in MCD scenarios'
radar_chart.x_labels =['Filter', 'Pump', 'Valve', 'Nozzle', 'Pipe']
radar_chart.add('SPC', [Algo_data_df.SPC_Model[0],Algo_data_df.SPC_Model[1],Algo_data_df.SPC_Model[2],Algo_data_df.SPC_Model[3],Algo_data_df.SPC_Model[4]])
radar_chart.add('Ensemble', [Algo_data_df.Ensemble_Classifiers[0],Algo_data_df.Ensemble_Classifiers[1],Algo_data_df.Ensemble_Classifiers[2],Algo_data_df.Ensemble_Classifiers[3],Algo_data_df.Ensemble_Classifiers[4]])
radar_chart.add('Neural_Network_Model', [Algo_data_df.Neural_Network_Model[0],Algo_data_df.Neural_Network_Model[1],Algo_data_df.Neural_Network_Model[2],Algo_data_df.Neural_Network_Model[3],Algo_data_df.Neural_Network_Model[4]])
radar_chart.add('Approx_Model', [Algo_data_df.Approximation_Model[0],Algo_data_df.Approximation_Model[1],Algo_data_df.Approximation_Model[2],Algo_data_df.Approximation_Model[3],Algo_data_df.Approximation_Model[4]]),
radar_chart.add('PINN_Model', [Algo_data_df.PINN_Model[0],Algo_data_df.PINN_Model[1],Algo_data_df.PINN_Model[2],Algo_data_df.PINN_Model[3],Algo_data_df.PINN_Model[4]]),
radar_chart.add('Ensemble_HDT_Model', [Algo_data_df.Ensemble_HDT_Model[0],Algo_data_df.Ensemble_HDT_Model[1],Algo_data_df.Ensemble_HDT_Model[2],Algo_data_df.Ensemble_HDT_Model[3],Algo_data_df.Ensemble_HDT_Model[4]])
radar_chart

# Means for Healthy condition. Just remember to concartinate all the data in the HC0 folder
#Healty
Mean_HC0_P1 = (HC0.P1).mean()

#Faulty Condition
FC1 = {'FC1_1': FC1_1, 'FC1_2': FC1_2, 'FC1_3': FC1_3, 'FC1_4': FC1_4, 'FC1_5': FC1_5}
FC2 = {'FC2_1': FC2_1, 'FC2_2': FC2_2, 'FC2_3': FC2_3, 'FC2_4': FC2_4, 'FC2_5': FC2_5}
FC3 = {'FC3_1': FC3_1, 'FC3_2': FC3_2, 'FC3_3': FC3_3, 'FC3_4': FC3_4, 'FC3_5': FC3_5}
FC4 = {'FC4_1': FC4_1, 'FC4_2': FC4_2, 'FC4_3': FC4_3, 'FC4_4': FC4_4, 'FC4_5': FC4_5}
FC5 = {'FC5_1': FC5_1, 'FC5_2': FC5_2, 'FC5_3': FC5_3, 'FC5_4': FC5_4, 'FC5_5': FC5_5}

# Calculate the means 
Mean_FC1_1_P1 = FC1['FC1_1']['P1'].mean()
Mean_FC2_1_P1 = FC1['FC1_1']['P1'].mean()

# Calculate the error using healthy condition as reference
Err_FC1_1_P1 = ((abs(Mean_HC0_P1-Mean_FC1_1_P1))/(Mean_HC0_P1))*100
Err_FC2_1_P1 = ((abs(Mean_HC0_P1-Mean_FC1_1_P1))/(Mean_HC0_P1))*100

# Create a Neural Network. See example below and update to suit what you want
# Generate some random data 
np.random.seed(0)
delta_p = np.random.rand(1000, 1)  # Delta Pressure
RPM = np.random.randint(0, 100, (1000, 1))  # Revolutions Per Minute
flow_rate = np.random.rand(1000, 1)  # Flow Rate

# Split the data into training and testing sets
delta_p_train, delta_p_test, RPM_train, RPM_test, flow_rate_train, flow_rate_test = train_test_split(delta_p, RPM, flow_rate, test_size=0.2, random_state=42)

# Define the architecture of the neural network
delta_p_input = Input(shape=(1,), name='delta_p_input')
RPM_input = Input(shape=(1,), name='RPM_input')

# Combine the inputs
combined = concatenate([delta_p_input, RPM_input])

# Hidden layers
hidden1 = Dense(32, activation='relu')(combined)
hidden2 = Dense(16, activation='relu')(hidden1)

# Output layer
output = Dense(1, activation='linear')(hidden2)

# Create the model
model = Model(inputs=[delta_p_input, RPM_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[mean_absolute_error, mean_absolute_percentage_error])

# Train the model
model.fit([delta_p_train, RPM_train], flow_rate_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, mae, mape = model.evaluate([delta_p_test, RPM_test], flow_rate_test)
print(f'Test loss: {loss}, MAE: {mae}, MAPE: {mape}')

