from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np

def preprocessing(path):
	df = pd.read_csv(path, delimiter=',', index_col=False)
	df.dataframeName = 'dataset.csv'

	X = df.iloc[:,1:]
	Y = df.iloc[:,0]
	l = ['complement'] * (121 - X.shape[1]) 

	for index,col in enumerate(l):
		X[col+str(index)] = 0

	X = X.values
	Y = Y.values

	Y_encoded = np.zeros((Y.shape[0], 3))
	for i,val in enumerate(Y):
		Y_encoded[i][int(val)] = 1

	indexes = np.arange(len(X))
	np.random.seed(42)
	np.random.shuffle(indexes)
	X, Y_encoded = X[indexes], Y_encoded[indexes]

	return X, Y_encoded

class SeqGen(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y