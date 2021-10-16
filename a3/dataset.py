import torch
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = []
        self._preprocess_data(data_path)

    def _preprocess_data(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = {}
        d["state"] = self.data[index][0]
        d["action"] = self.data[index][1]

        return d
