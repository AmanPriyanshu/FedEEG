import torch
import pandas as pd

class EEG_DataLoader(torch.utils.data.Dataset):
	
	def __init__(self, file_path):
		self.data = pd.read_csv(file_path)
		self.data = self.data.values
		
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, index):
		eeg_data = torch.from_numpy(self.data[index][1:])
		eeg_type = torch.tensor(int(self.data[index][0]))
		return eeg_data, eeg_type