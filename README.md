# Wafer Map Classification Using Deep Learning

## Project Description
This project is for wafer map classification tasks on the WM-811K dataset. It consists of various deep learning models, including Convolutional Neural Networks (CNN), Deep Convolutional Neural Networks (DCNN), DenseNet121, MobileNet, ResNet18, ResNet50, ResNet101, and ResNet152. The pipeline includes preprocessing, training, and result visualization stages.

## Dataset
The dataset used in this project is the WM-811K dataset. The dataset consists of wafer maps with patterns and failures collected from real-world production environments. 

To use this dataset with our project, follow these steps:
1. Download the `LSWMD.pkl` file.
2. Place it in the `dataset` directory.

## Directory Structure
- **dataset**: Place the `LSWMD.pkl` here. After running the preprocessing script, the preprocessed data will also be saved in this folder.
- **models**: This folder should contain the model files (`cnn.py`, `dcnn.py`, `densenet121.py`, `mobilenet.py`, `resnet18.py`, `resnet50.py`, `resnet101.py`, `resnet152.py`). Each file defines a different deep learning model architecture.
- **results**: The results from the models will be saved in pickle format in this directory.

## Preprocessing

Data preprocessing is a crucial stage in any machine learning project as it transforms raw data into an understandable and suitable format for further processing and model training. In the `preprocessing.ipynb` Jupyter notebook, we describe this process for our wafer map dataset in detail. 

### Data Cleaning

The initial step in our preprocessing pipeline involves cleaning the data, which includes removing or correcting wrong, corrupted, or inaccurately recorded data. This helps ensure the quality of the dataset used for further processing.

### Data Transformation

Once the data is cleaned, it needs to be transformed into a suitable format for training our machine learning model. This transformation process includes several key steps:

- **Resizing**: Given that the wafer maps in the dataset have varying dimensions, they are resized to a uniform size. This ensures that our model receives consistently shaped input.

- **Flattening**: Post resizing, the 2D wafer map arrays are flattened into 1D, which is a prerequisite for the upcoming oversampling process.

- **Balancing Classes**: We balance the dataset to mitigate any skewness that might exist. For the training data, this involves oversampling the minority classes using Synthetic Minority Over-sampling Technique (SMOTE). For the test data, it involves undersampling the majority class to align with the count of other classes.

- **Reshaping**: Following the balancing process, the wafer map data is reshaped back into its original 2D form to maintain the structural information of the wafer maps.

### Data storage

The preprocessed data is stored for later use in the model training stage.

## Training
The training stage is handled by `Train.ipynb`. It uses the models mentioned above to train on the preprocessed data. This involves configuring the models, setting the training parameters, and running the training process.

## Results
The `result.py` script reads the results from the pickle files and displays them. This allows you to easily view and compare the performance of the different models.

## How to Use
1. Place the `LSWMD.pkl` file in the `dataset` directory.
2. Run `preprocessing.ipynb` to preprocess the data.
3. Run `Train.ipynb` to train the models. Ensure that the corresponding model files are in the `models` directory.
4. After training, the results will be saved in the `results` directory.
5. Run `result.py` to view the results.

## Requirements
List the required libraries and their versions here. For example:
- Python 3.7.9
- PyTorch 2.0.0.dev20230125+cu118
- torchvision 0.15.0.dev20230127+cu118
- NumPy 1.21.6
- pandas 1.3.5
- Matplotlib 3.5.3
- tqdm 4.64.0

