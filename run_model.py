"""
CIS 472 Machine Learning Final Project

Author: Krishna Patel
Last Updated: 06/10/2023

Description: 
"""


from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score


def train_and_eval(arguments, model, train_data_loader, test_data_loader, eval_data_loader):
	optimizer = arguments.optim(model.parameters(), arguments.learning_rate)
	device = arguments.device
	for epoch in range(arguments.epochs):
		print(f"--- Epoch {epoch + 1}/{arguments.epochs} ---")
		running_loss = 0.0
		model.train()
		train_targets, train_preds = [], []
		for images, labels in tqdm(train_data_loader,  desc = "train"):
			image = images.to(device)
			label = labels.to(device)

			optimizer.zero_grad()
			output = model(image)
			predictions = torch.argmax(output, dim=-1)
			train_preds += predictions.detach().cpu().numpy().tolist()
			train_targets += label.detach().cpu().numpy().tolist()
			loss = arguments.loss_fn()(output, label)
			loss.backward()
			optimizer.step()
			
			#running_loss += loss.item()
		model.eval()
		dev_targets, dev_preds = [], []
		for images, labels in tqdm(test_data_loader,  desc = "test"):
		
			
			image = images.to(device)
			label = labels.to(device)

			optimizer.zero_grad()
			output = model(image)
			predictions = torch.argmax(output, dim=-1)
			loss = arguments.loss_fn()(output, label)
			loss.backward()
			optimizer.step()
			dev_preds += predictions.detach().cpu().numpy().tolist()
			dev_targets += label.detach().cpu().numpy().tolist()

		test_acc = accuracy_score(dev_preds, dev_targets)
		train_acc  = accuracy_score(train_preds, train_targets)
		#epoch_loss = running_loss / len(train_data_loader)
		print(f"train acc: {train_acc}")
		print(f"test acc : {test_acc}")

	print("===VALIDATION RUN===")
	model.eval()
	val_targets, val_preds = [], []
	for images, labels in tqdm(eval_data_loader,  desc = "validation"):
		image = images.to(device)
		label = labels.to(device)

		optimizer.zero_grad()
		output = model(image)
		predictions = torch.argmax(output, dim=-1)
		loss = arguments.loss_fn()(output, label)
		loss.backward()
		optimizer.step()
		val_preds += predictions.detach().cpu().numpy().tolist()
		val_targets += label.detach().cpu().numpy().tolist()

	valid_acc = accuracy_score(val_preds, val_targets)
	print(f"valid acc : {valid_acc}")
