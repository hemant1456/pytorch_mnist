# pytorch_mnist
This repository provides code and instructions to train a convolutional neural network (CNN) on the MNIST dataset using PyTorch.

Table of Contents
Dependencies
Model Architecture
Utility Functions
Running the Model
Instructions to Run
Results
Notes
Dependencies <a name="dependencies"></a>
This project requires the following Python libraries:

torch
torchvision
matplotlib
numpy
tqdm
torchsummary
You can install them using pip:

shell
Copy code
pip install torch torchvision matplotlib numpy tqdm torchsummary
Model Architecture <a name="model-architecture"></a>
The model is a simple yet effective convolutional neural network (CNN) for digit recognition. It contains four convolutional layers, each followed by a ReLU activation function, and two fully connected layers at the end. The model uses max pooling and flattening before passing the result to the fully connected layers. The output of the final layer is a log softmax output, which is suitable for multi-class classification problems like digit recognition.

Utility Functions <a name="utility-functions"></a>
This file contains utility functions for training, testing, and visualizing the model's performance.

GetCorrectPredCount: Counts the number of correct predictions in a batch.
train_model: Trains the model for one epoch, calculates the loss and updates the weights.
test_model: Evaluates the model on the test data and calculates the test loss and accuracy.
plot_images: Plots a batch of images with their corresponding labels.
plot_losses: Plots the training and test losses and accuracies over epochs.
Running the Model <a name="running-the-model"></a>
This script loads the MNIST dataset, applies transformations, defines the model, optimizer, loss function, and trains the model for a given number of epochs.

Instructions to Run <a name="instructions-to-run"></a>
To train the model, navigate to the directory containing the files and run the main.py script using Python:

shell
Copy code
python main.py
Results <a name="results"></a>
The training and testing losses and accuracies will be plotted after the training is done. You can inspect these plots to understand the model's performance.

Notes <a name="notes"></a>
This is a simple project intended for educational purposes. As such, the model architecture, training, and testing procedures are relatively simple and may not give state-of-the-art results on more complex datasets.

