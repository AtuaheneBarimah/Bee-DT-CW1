# Import Python Lib. You can add if required
!pip install --upgrade keras scikit-learn
!pip install cairosvg
!pip install pygal
import cairosvg
import os
import pygal
from pathlib import Path
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
columns_to_drop = ['X_Value',	'Untitled',	'Comment']

class KeyDataFrame(pd.DataFrame):
    def __repr__(self):
        return super().__repr__()

def process_lvm_file(file_path, week_name):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    variable_name = '_'.join(file_name.split('_')[:2])
    df = pd.read_csv(file_path, delimiter='\t', skiprows=23, header=None)
    column_names = ['X_Value','Pre-Filter Pressure Transducer',	'Post Filter Pressure Transducer',	'Pre Valve Pressure Transducer','Post Valve Pressure Transducer','Main Tank Flow Meter','Sump Tank Flow Meter',	'End Pressure','Untitled','Untitled 1',	'Untitled 2',	'Untitled 3',	'Untitled 4',	'Untitled 5','Comment'] #Columns to keep using the drop columns feature
    df.columns = column_names
    df = df.drop(columns=columns_to_drop)
    data_frames[week_name][variable_name] = KeyDataFrame(df)

class DataFrameDict(dict):
    def __missing__(self, key):
        self[key] = {}
        return self[key]
data_frames = DataFrameDict()
week_folders = ['/content/drive/MyDrive/CW1/FC0',
                '/content/drive/MyDrive/CW1/FC1',
                '/content/drive/MyDrive/CW1/FC2',
                '/content/drive/MyDrive/CW1/FC3',
                '/content/drive/MyDrive/CW1/FC4',
                '/content/drive/MyDrive/CW1/FC5']

for week_folder in week_folders:
    week_name = Path(week_folder).name
    for root, dirs, files in os.walk(week_folder):
        for file in files:
            if file.endswith(".lvm"):
                file_path = os.path.join(root, file)
                process_lvm_file(file_path, week_name)

globals().update(data_frames)


def replace_and_modify_columns(df, x):
    df = df.rename(columns={'Pre-Filter Pressure Transducer': 'P1_bar',
                            'Post Filter Pressure Transducer': 'P2_bar',
                            'Pre Valve Pressure Transducer': 'P3_bar',
                            'Post Valve Pressure Transducer': 'P4_bar',
                            'Untitled': 'RPM',
                            'Main Tank Flow Meter': 'F1',
                            'Untitled 1': 'DPV_1',
                            'Untitled 2': 'DPV_2',
                            'Untitled 3': 'DPV_3',
                            'Untitled 4': 'DPV_4',
                            'Untitled 5': 'DPV_5',
                            'Sump Tank Flow Meter': 'F2',
                            'End Pressure': 'P5_bar'})

    numeric_columns = ['P3_bar', 'P4_bar', 'P5_bar']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    df['P4_bar'] += x
    df['P3_bar'] += x
    df['P5_bar'] += x
    return dfdef replace_and_modify_columns(df, x):
    df = df.rename(columns={'Pre-Filter Pressure Transducer': 'P1_bar',
                            'Post Filter Pressure Transducer': 'P2_bar',
                            'Pre Valve Pressure Transducer': 'P3_bar',
                            'Post Valve Pressure Transducer': 'P4_bar',
                            'Untitled': 'RPM',
                            'Main Tank Flow Meter': 'F1',
                            'Untitled 1': 'DPV_1',
                            'Untitled 2': 'DPV_2',
                            'Untitled 3': 'DPV_3',
                            'Untitled 4': 'DPV_4',
                            'Untitled 5': 'DPV_5',
                            'Sump Tank Flow Meter': 'F2',
                            'End Pressure': 'P5_bar'})

    numeric_columns = ['P3_bar', 'P4_bar', 'P5_bar']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    df['P4_bar'] += x
    df['P3_bar'] += x
    df['P5_bar'] += x
    return df

HS_0 =  (pd.concat([(FC0['FC0_100']),(FC0['FC0_101'])]))[1:]
HS_1 =  (pd.concat([(FC1['FC1_100']),(FC1['FC1_101'])]))[1:]

FC1_1 =replace_and_modify_columns((HS_0),1.0101) # Example use assuming x = 1.0101 and HS_0 is your Data 

# Plot Process Data and Variance
QSS=FC1_1
A = QSS.P1_bar
B = QSS.P2_bar
C = QSS.P3_bar
D = QSS.P4_bar
E = QSS.P5_bar
F = QSS.F1

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of Samples')
ax1.set_ylabel('Pressure (bar)', color=color)
ax1.plot(A, color='red', label='P1')
ax1.plot(B, color='blue', label='P2')
ax1.plot(C, color='green', label='P3')
ax1.plot(D, color='orange', label='P4')
ax1.plot(E, color='purple', label='P5')
ax1.tick_params(axis='y', labelcolor=color)


ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Flow (l/min)', color=color)
ax2.plot(F, color='magenta', label='Flow(l/min)')
ax2.tick_params(axis='y', labelcolor=color)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax1.legend(lines, labels)

fig.tight_layout()
plt.title('Graph of Pressure Signature Vs Flow')
plt.show()

Sdata = {
    'P1_bar': A,
    'P2_bar': B,
    'P3_bar': C,
    'P4_bar': D,
    'P5_bar': E,
    'F1_lpermin': F
}


Sest = pd.DataFrame(Sdata)

column_variances = {}
for column in Sest.columns:
    column_variances[column] = Sest[column].var()

max_variance_column = max(column_variances, key=column_variances.get)

print("Parameter with the highest variance:", max_variance_column)

plt.bar(column_variances.keys(), column_variances.values())
plt.xlabel("System Variables")
plt.ylabel("Variance")
plt.title("Variances in System Parameters for FC25_700")
plt.xticks(rotation=90)
plt.show()

# Plot Histogram for various parameters
S1 = (FC0_1)
S2 = (FC1_1)
S3 = (FC2_1)
S4 = (FC3_1)
num_bins = 50
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


def Mean_Check(H,FM1,FM2):
  P1_1 = ((abs(((H.mean()).P1_bar)-((FM1.mean()).P1_bar)))/((H.mean()).P1_bar))*100
  P2_1 = ((abs(((H.mean()).P2_bar)-((FM1.mean()).P2_bar)))/((H.mean()).P2_bar))*100
  P3_1 = ((abs(((H.mean()).P3_bar)-((FM1.mean()).P3_bar)))/((H.mean()).P3_bar))*100
  P4_1 = ((abs(((H.mean()).P4_bar)-((FM1.mean()).P4_bar)))/((H.mean()).P4_bar))*100
  P5_1 = ((abs(((H.mean()).P5_bar)-((FM1.mean()).P5_bar)))/((H.mean()).P5_bar))*100
  F1_1 = ((abs(((H.mean()).F1)-((FM1.mean()).F1)))/((H.mean()).F1))*100

  P1_2 = ((abs(((H.mean()).P1_bar)-((FM2.mean()).P1_bar)))/((H.mean()).P1_bar))*100
  P2_2 = ((abs(((H.mean()).P2_bar)-((FM2.mean()).P2_bar)))/((H.mean()).P2_bar))*100
  P3_2 = ((abs(((H.mean()).P3_bar)-((FM2.mean()).P3_bar)))/((H.mean()).P3_bar))*100
  P4_2 = ((abs(((H.mean()).P4_bar)-((FM2.mean()).P4_bar)))/((H.mean()).P4_bar))*100
  P5_2 = ((abs(((H.mean()).P5_bar)-((FM2.mean()).P5_bar)))/((H.mean()).P5_bar))*100
  F1_2 = ((abs(((H.mean()).F1)-((FM2.mean()).F1)))/((H.mean()).F1))*100

  Algo_data = {
    'Variable': ['P1_bar', 'P2_bar', 'P3_bar', 'P4_bar', 'P5_bar','F1'],
    'FM1': [P1_1, P2_1, P3_1, P4_1, P5_1,F1_1],
    'FM2': [P1_2, P2_2, P3_2, P4_2, P5_2,F1_2]
  }
  Algo_data_df = pd.DataFrame(Algo_data)
  Algo_data_df

  return Algo_data_df

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

