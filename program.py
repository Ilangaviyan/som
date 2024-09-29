import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('path_to_your_file/purchase_data_exe.csv')

# Clean the dataset by dropping unnecessary columns
dataset_cleaned = dataset.drop(columns=['Unnamed: 7', 'date', 'customer_id', 'payment_method'])

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(dataset_cleaned)

# Initialize the SOM (10x10 grid)
som = MiniSom(x=10, y=10, input_len=data_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(data_scaled)

# Train the SOM
som.train_random(data_scaled, 100)

# Visualize the SOM distance map
plt.figure(figsize=(10, 7))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Distance map as background
plt.colorbar()

# Plot markers for each data point
for i, x in enumerate(data_scaled):
    w = som.winner(x)  # Get the winning node
    plt.plot(w[0] + 0.5, w[1] + 0.5, 'o', markeredgecolor='black',
             markerfacecolor='None', markersize=10, markeredgewidth=2)

plt.title('Customer Segmentation Using SOM')
plt.show()
