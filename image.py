from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def generate_image(arr, name):
	plt.cla()
	for client_id, row in enumerate(arr):
		plt.plot(np.arange(len(row)), row, label='client_id_'+str(client_id))
	plt.legend()
	if 'acc' in name:
		plt.ylim([-.01, 1.01])
	plt.savefig(name)

if __name__ == '__main__':
	history = pd.read_csv('history.csv')
	history = history.values

	loss = [history.T[i] for i in range(0, history.shape[1], 2)]
	acc = [history.T[i] for i in range(1, history.shape[1], 2)]

	generate_image(loss, './images/loss.png')
	generate_image(acc, './images/acc.png')