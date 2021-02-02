import torch

def model_generator():
	model = torch.nn.Sequential(
		torch.nn.LSTM(112, 64, 2),
		torch.nn.Flatten(),
		#torch.nn.Linear(112, 84),
		#torch.nn.ReLU(),
		torch.nn.Linear(84, 64),
		torch.nn.ReLU(),
		torch.nn.Linear(64, 42),
		torch.nn.ReLU(),
		torch.nn.Linear(42, 32),
		torch.nn.ReLU(),
		torch.nn.Linear(32, 8),
		torch.nn.ReLU(),
		torch.nn.Linear(8, 3),
		)
	return model