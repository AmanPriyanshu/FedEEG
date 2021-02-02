import torch
from code.model import model_generator
from code.eeg_dataloader import EEG_DataLoader
from code.helpers import get_weights, set_weights, average_weights
from tqdm import trange
import pandas as pd
import time

class SingleTrainer:
	def __init__(self, path='./dataset_hand_movement/user_a.csv', batch_size=32):
		self.batch_size = batch_size
		self.train_dataloader = self.dataloader(path)
		self.model = model_generator()
		self.criterion = torch.nn.CrossEntropyLoss()

	def train_single_epoch(self, model_weights, epoch=1):
		self.model = set_weights(self.model, model_weights)
		optimizer = torch.optim.Adam(self.model.parameters())
		dataset_iter = iter(self.train_dataloader)
		progression = trange(len(dataset_iter))
		running_loss = 0
		running_accuracy = 0
		for batch_id in progression:
			data, label = dataset_iter.next()
			data = data.float().unsqueeze(0)

			optimizer.zero_grad()
			output = self.model(data)
			loss = self.criterion(output, label)
			loss.backward()
			optimizer.step()
			output = torch.argmax(output, 1)
			accuracy = torch.sum(output == label)/label.shape[0]

			running_loss += loss.item()
			running_accuracy += accuracy.item()

			progression.set_description(str({'loss': round(running_loss/(batch_id+1), 4), 'accuracy': round(running_accuracy/(batch_id+1), 4), 'epoch': epoch}))
			
		return get_weights(self.model)

	def train(self, model_weights, epochs=3):
		for epoch in range(epochs):
			model_weights = self.train_single_epoch(model_weights, epoch)
		return model_weights
			
	def dataloader(self, path):
		train_dataset = EEG_DataLoader(path)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
		return train_loader

if __name__ == '__main__':
	st = SingleTrainer(batch_size=512)
	model_weights = get_weights(st.model)
	st.train(model_weights)