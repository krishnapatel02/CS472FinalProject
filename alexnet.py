"""
CIS 472 Machine Learning Final Project

Author: Linnea Gilius and Krishna Patel
Credits: https://en.wikipedia.org/wiki/AlexNet
Last Updated: 06/10/2023

Description: contains two implementations of AlexNet and an implementation of LeNet
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
	def __init__(self, args):
		super(AlexNet, self).__init__()
		self.args = args
		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(96)
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(96, 256, kernel_size=5, stride=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(256)
		)
		self.layer3 = nn.Sequential(
			nn.Conv2d(256, 384, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(384, 384, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(256),
			nn.Flatten()
		)
		self.linearLayer1 = nn.Sequential(
			nn.Linear(1024, 4096),
			nn.Dropout(args.dropout),
			nn.ReLU(),
			# nn.MaxPool2d(kernel_size = 3, stride = 2)
		)
		self.linearLayer2 = nn.Sequential(
			nn.Linear(4096, 4096),
			nn.Dropout(args.dropout),
			nn.ReLU()
		)
		self.linearLayer3 = nn.Sequential(
			nn.Dropout(args.dropout),
			nn.Linear(4096, 5)
		)
		
	def forward(self, x):
		output = self.layer1(x)
		output = self.layer2(output)
		output = self.layer3(output)
		#output = output.reshape(output.size(0), -1)
		output = self.linearLayer1(output)
		output = self.linearLayer2(output)
		for i in range(self.args.hidden_layers):
			output = self.linearLayer2(output)
		output = self.linearLayer3(output)
		return output


class AlexNet_Modified(nn.Module):
	def __init__(self, args):
		super(AlexNet_Modified, self).__init__()
		self.args = args
		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(96)
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(96,256, kernel_size=5, stride=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(256)
		)
		self.layer3 = nn.Sequential(
			nn.Conv2d(256, 384, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(384, 384,kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.BatchNorm2d(256),
			nn.Flatten()
		)
		self.linearLayer1 = nn.Sequential(
			nn.Dropout(args.dropout),
			nn.Linear(1024, 5)
			#nn.MaxPool2d(kernel_size = 3, stride = 2)
		)
		
	def forward(self, x):
		output = self.layer1(x)
		output = self.layer2(output)
		output = self.layer3(output)
		#output = output.reshape(output.size(0), -1)
		output = self.linearLayer1(output)
		return output


class LeNet(nn.Module):
	def __init__(self, args):
		super(LeNet, self).__init__()
		self.args = args
		
		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 6, kernel_size = 5, padding = 2),
			nn.Sigmoid(),
			nn.MaxPool2d(kernel_size = 2, stride=2),
			nn.BatchNorm2d(6)
		)

		self.layer2 = nn.Sequential(
			nn.Conv2d(6, 16, kernel_size = 5),
			nn.Sigmoid(),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.BatchNorm2d(16)
		)

		self.linear_layer1 = nn.Sequential(
			nn.Linear(61504, 120),
			nn.Sigmoid(),
		)

		self.linear_layer2 = nn.Sequential(
			nn.Linear(120, 84),
			nn.Sigmoid()
		)

		self.linear_layer3 = nn.Sequential(
			nn.Linear(84, 5)
		)
		
	def forward(self, x):
		output = self.layer1(x)
		output = self.layer2(output)
		output = output.view(output.size(0), -1)
		output = self.linear_layer1(output)
		output = self.linear_layer2(output)
		output = self.linear_layer3(output)
		return output
