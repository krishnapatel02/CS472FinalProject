"""
CIS 472 Machine Learning Final Project

Authors: Linnea Gilius and Krishna Patel
Last Updated: 06/10/2023

Description: 
"""


import torch.optim as optim
import torchvision.models as models

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from make_datasets import *
from alexnet import *
from resnet import *
from run_model import train_and_eval


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class args():
	def __init__(self, epochs, dropout, batch_size, lr, optim, loss_fn, hidden_layers):
		self.epochs = epochs
		self.dropout = dropout
		self.num_classes = 5
		self.batch_size = batch_size
		self.learning_rate = lr
		self.optim = optim
		self.hidden_layers = hidden_layers
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(self.device)
		self.loss_fn = loss_fn
	
	def __str__(self) -> str:
		return f"Epochs: {self.epochs}\nDropout: {self.dropout}\nBatch Size: {self.batch_size}\nlearn_rate: {self.learning_rate}\noptim: {self.optim} \nloss_fn: {self.loss_fn}" 


def runTests(arguments):
	print(arguments)
	train_data_loader = DataLoader(dataset, batch_size=arguments.batch_size, sampler=train_sampler)
	test_data_loader = DataLoader(dataset, batch_size=arguments.batch_size, sampler=test_sampler)
	validation_data_loader = DataLoader(dataset, batch_size=arguments.batch_size, sampler=validation_sampler)
	
	print("Testing AlexNet, modified")
	model = AlexNet_Modified(arguments)
	# print(model)
	model.to(device)
	train_and_eval(arguments, model, train_data_loader, test_data_loader, validation_data_loader)
	
	print("Testing AlexNet with Perceptron")
	model = AlexNet(arguments)
	model.to(device)
	train_and_eval(arguments, model, train_data_loader, test_data_loader, validation_data_loader)
	
	print("Testing PyTorch's AlexNet")
	model = models.AlexNet()
	model.to(device)
	train_and_eval(arguments, model, train_data_loader, test_data_loader, validation_data_loader)

	print("Testing LeNet")
	model = LeNet(arguments)
	model.to(device)
	train_and_eval(arguments, model, train_data_loader, test_data_loader, validation_data_loader)

	print("Change to 10 Epochs")
	arguments.epochs = 10

	print("Testing resnet18")
	model = resnet18(arguments)
	model.to(device)
	train_and_eval(arguments, model, train_data_loader, test_data_loader, validation_data_loader)

	print("Testing resnet50")
	model = resnet18(arguments)
	model.to(device)
	train_and_eval(arguments, model, train_data_loader, test_data_loader, validation_data_loader)

	print("Testing built in resnet18")
	model = models.resnet18()
	model.to(device)
	train_and_eval(arguments, model, train_data_loader, test_data_loader, validation_data_loader)