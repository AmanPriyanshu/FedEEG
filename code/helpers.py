import torch
import numpy as np
import pandas as pd

def get_weights(model, dtype=np.float32):
	weights = []
	for layer in model:
		try:
			weights.append([layer.weight.detach().numpy().astype(dtype), layer.bias.detach().numpy().astype(dtype)])
		except:
			continue
	return np.array(weights)

def set_weights(model, weights):
	index = 0
	for layer_no, layer in enumerate(model):
		try:
			_ = model[layer_no].weight
			model[layer_no].weight = torch.nn.Parameter(weights[index][0])
			model[layer_no].bias = torch.nn.Parameter(weights[index][1])
			index += 1
		except:
			continue
	return model

def average_weights(all_weights):
	all_weights = np.array(all_weights)
	all_weights = np.mean(all_weights, axis=0)
	all_weights = [[torch.from_numpy(i[0].astype(np.float32)), torch.from_numpy(i[1].astype(np.float32))] for i in all_weights]
	return all_weights