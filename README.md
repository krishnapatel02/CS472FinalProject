# CS 472 Final Project

<br>

### Description

This project based on a contest on Kaggle: Making Graphs Accessible
([link](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/)),
and the dataset is taken from Kaggle ([link](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/data)). 
The implementation is done using PyTorch and other machine learning libraries.


<br>

### Files
`alexnet.py` contains two implementations of AlexNet and one implementation of LeNet <br>
`main.py` creates the dataloaders used for training and contains the argument class used to adjust hyperparameters used in training. Models are initialized and the argument class is passed along with the model to run_model.py for testing. <br>
`make_datasets.py` goes through the image and annotation directories to create a dataset which is used to randomly populate three dataloaders. The images are processed with torchvision and the chart type is extracted from the annotations and turned into an int. <br>
`resnet.py` contains an implementation of ResNet, includes ResNet-18, ResNet-34, and ResNet-50. <br>
`run_model.py` contains graphing and training code, called by the main file for each model being tested <br>

<br>

### Contributors
Linnea Gilius and Krishna Patel
