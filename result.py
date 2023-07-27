import pickle
import os
import matplotlib.pyplot as plt


# Function to load data from pickle files
def load_data_from_pkl(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data


# Set the path to the results folder
results_folder = 'results'

# Get the list of all pickle files in the results folder
pkl_files = [os.path.join(results_folder, f) for f in os.listdir(results_folder) if f.endswith('.pkl')]

# Initialize empty dictionaries to store the data from all models
all_time_per_epoch = {}
all_loss_values_train = {}
all_loss_values_test = {}
all_accuracies_train = {}
all_accuracies_test = {}

# Read data from all pickle files and store them in the dictionaries
for file_name in pkl_files:
    model_name = file_name[8:-4]  # Remove the '.pkl' extension to get the model name
    data = load_data_from_pkl(file_name)

    all_time_per_epoch[model_name] = data['time_per_epoch']
    all_loss_values_train[model_name] = data['loss_values'][0]
    all_loss_values_test[model_name] = data['loss_values'][1]
    all_accuracies_train[model_name] = data['accuracy'][0]
    all_accuracies_test[model_name] = data['accuracy'][1]

print(all_loss_values_test)
print(all_accuracies_test)


# Function to plot the data
def plot_data(data, ylabel, title):
    plt.figure(figsize=(12,10))
    for model_name, values in data.items():
        plt.plot(values, label=model_name)
    plt.xlabel('Epoch', fontdict={'size': 18})
    plt.ylabel(ylabel, fontdict={'size': 18})
    plt.title(title, fontdict={'size': 18})
    plt.legend(prop={'size': 18})
    plt.show()


# Plot the figures
plot_data(all_time_per_epoch, 'Time per Epoch', 'Comparison of Time per Epoch')
plot_data(all_loss_values_train, 'Train Loss', 'Comparison of Training Loss')
plot_data(all_loss_values_test, 'Test Loss', 'Comparison of Testing Loss')
plot_data(all_accuracies_train, 'Train Accuracy', 'Comparison of Training Accuracy')
plot_data(all_accuracies_test, 'Test Accuracy', 'Comparison of Testing Accuracy')
