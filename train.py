import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from code.model import get_model
from code.train_single_client import train_single_client
from code.dataloader import preprocessing
import pandas as pd

def train(epochs=5):

	x, _ = preprocessing('./dataset_hand_movement/user_a.csv')
	x_mean = np.mean(x, axis=0)
	x_std = np.std(x, axis=0)

	model = get_model()
	aggregate_weights = model.get_weights()
	client_performances = []
	test_performances = []
	for _ in range(epochs):
		all_weights = []
		performances = []
		for client in ['./dataset_hand_movement/'+i+'.csv' for i in ['user_a', 'user_b', 'user_c', 'user_d']]:
			weights, history = train_single_client(aggregate_weights, client, x_mean, x_std)
			loss = history.history['val_loss'][-1]
			accuracy = history.history['val_accuracy'][-1]
			performances.append([loss, accuracy])
			all_weights.append(weights)
		aggregate_weights = np.stack(all_weights)
		aggregate_weights = np.mean(aggregate_weights, axis=0)
		model.set_weights(aggregate_weights)
		client_performances.append(performances)

	client_performances = np.array(client_performances)
	names = np.array([['client_id_'+str(client_id)+'_loss', 'client_id_'+str(client_id)+'_acc'] for client_id in range(client_performances.shape[1])]).flatten()
	client_performances = client_performances.reshape(client_performances.shape[0], client_performances.shape[1]*client_performances.shape[2])
	client_performances = pd.DataFrame(client_performances)
	client_performances.columns = [i for i in names]

	return aggregate_weights, client_performances

if __name__ == '__main__':
	weights, performances = train()
	model = get_model()
	model.set_weights(weights)
	model.save('trained_model.h5')
	performances.to_csv('./history.csv', index=False)