"""
CIS 472 Machine Learning Final Project

Author: Linnea Gilius
Credits: https://en.wikipedia.org/wiki/Residual_neural_network 
Last Updated: 06/10/2023

Description: contains an implementation of ResNet
"""


import torch
import torch.nn as  nn
import torch.nn.functional as F


class Basic_Block(nn.Module):
	expansion = 1

	def __init__(self, in_channels, out_channels, stride = 1):
		super(Basic_Block, self).__init__()

		self.convolution1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
		self.batch_norm1 = nn.BatchNorm2d(out_channels)
		self.convolution2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
		self.batch_norm2 = nn.BatchNorm2d(out_channels)	
		self.relu = nn.ReLU()
		self.stride = stride

	def forward(self, x):

		identity = x
		output = self.convolution1(x)
		output = self.batch_norm1(output)
		output = self.relu(output)

		output = self.convolution2(output)
		output = self.batch_norm2(output)
		output = output + identity
		output = self.relu(output)

		return output


class Bottleneck_Block(nn.Module):
	expansion = 4

	def __init__(self, in_channels, out_channels, stride = 1):
		super(Bottleneck_Block, self).__init__()

		self.convolution1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
		self.batch_norm1 = nn.BatchNorm2d(out_channels)
		self.convolution2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
		self.batch_norm2 = nn.BatchNorm2d(out_channels)
		self.convolution3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size = 1, stride = 1, padding = 0, bias = False)
		self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)
		self.relu = nn.ReLU()
		self.stride = stride

	def forward(self, x):

		identity = x
		output = self.convolution1(x)
		output = self.batch_norm1(output)
		output = self.relu(output)

		output = self.convolution2(output)
		output = self.batch_norm2(output)
		output = self.relu(output)

		output = self.convolution3(output)
		output = self.batch_norm3(output)
		output = output + identity
		output = self.relu(output)

		return output


class ResNet(nn.Module):
	def __init__(self, block, layer_list, num_classes):
		super(ResNet, self).__init__()
		
		self.in_channels = 64
		self.convolution = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
		self.batch_norm = nn.BatchNorm2d(self.in_channels)
		self.relu = nn.ReLU()

		self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		self.max_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
		self.fully_connected = nn.Linear(512 * block.expansion, num_classes)
		
		self.layer1 = self.build_layer(block, 64, layer_list[0])
		self.layer2 = self.build_layer(block, 128, layer_list[1], stride = 2)
		self.layer3 = self.build_layer(block, 256, layer_list[2], stride = 2)
		self.layer4 = self.build_layer(block, 512, layer_list[3], stride = 2)
		
	def build_layer(self, block, planes, blocks, stride = 1):
		layers = []
		layers.append(block(self.in_channels, planes, stride))
		self.in_channels = planes * block.expansion

		for i in range(blocks - 1):
			layers.append(block(self.in_channels, planes))
			
		return nn.Sequential(*layers)

	def forward(self, x):

		output = self.convolution(x)
		output = self.batch_norm(output)
		output = self.relu(output)
		output = self.max_pool(output)

		output = self.layer1(output)
		output = self.layer2(output)
		output = self.layer3(output)
		output = self.layer4(output)

		output = self.avgpool(output)
		output = torch.flatten(output, 1)
		output = self.fully_connected(output)

		return output
