import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from code.model import get_model
from code.dataloader import preprocessing, SeqGen
import numpy as np

def train_model(model, x, y, x_mean, x_std, epochs=3, split=0.8):
	x_train, y_train, x_test, y_test = x[:int(split*len(x))], y[:int(split*len(x))], x[int(split*len(x)):], y[int(split*len(x)):]

	x_train_standard, x_test_standard = (x_train - x_mean)/(x_std+1e-5), (x_test - x_mean)/(x_std+1e-5)
	x_train_standard, x_test_standard = x_train_standard.reshape((x_train_standard.shape[0], 11, 11)), x_test_standard.reshape((x_test_standard.shape[0], 11, 11))

	return model.fit(SeqGen(x_train_standard,y_train,batch_size=12), validation_data=(x_test_standard,y_test), epochs=epochs, verbose=1)

def train_single_client(model_weights, path, x_mean, x_std):
	x, y = preprocessing(path)
	model = get_model()
	if model_weights is not None:
		model.set_weights(model_weights)
	history = train_model(model, x, y, x_mean, x_std)
	return np.array(model.get_weights()), history

if __name__ == '__main__':
	train_single_client()
	