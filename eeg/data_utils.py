import os
import torch
import random
import numpy as np
from typing import List
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    task_num = 14
    p = 0.9  # partition ratio of train and test set from full dataset

    def __init__(self, data_dir: str, load_attr: str = None, train: bool = True, **kwargs):
        self._data_dir = os.path.abspath(data_dir) if not os.path.isabs(data_dir) else data_dir
        self._load_attr = f'_load_{load_attr}'
        self._train = train
        self._meta_data = []

        # load data based on different attribute
        if hasattr(self, self._load_attr):
            load_func = getattr(self, self._load_attr)
            load_func(**kwargs)
        else:
            print(f"Attribute '{load_attr}' not found in EEGDataset.")

        random.shuffle(self._meta_data)
        if self._train:
            self._meta_data = self._meta_data[:int(self.p * len(self._meta_data))]
        else:
            self._meta_data = self._meta_data[int(self.p * len(self._meta_data)):]

    def __len__(self):
        return len(self._meta_data)

    def __getitem__(self, index: int):
        data_path, label = self._meta_data[index]
        data = np.load(data_path)
        return torch.Tensor(data), torch.Tensor(label)
    
    # load full subjects and full tasks
    def _load_full(self):
        for subject in os.listdir(self._data_dir):
            for task in os.listdir(os.path.join(self._data_dir, subject)):
                for event in os.listdir(os.path.join(self._data_dir, subject, task)):
                    label = int(event.split('_')[0])
                    self._meta_data.append((os.path.join(self._data_dir, subject, task, event), label))

    # load full tasks of some subjects
    def _load_subject(self, subjects: List[int]):
        self._load_task(subjects, list(range(1, self.task_num + 1)))

    # load some tasks of some subjects
    def _load_task(self, subjects: List[int], tasks: List[int]):
        for subject in subjects:
            for task in tasks:
                path_dir = f'{self._data_dir}/{subject:03d}/{task:02d}'
                if os.path.exists(path_dir):
                    for event in os.listdir(path_dir):
                        label = int(event.split('_')[0])
                        self._meta_data.append((os.path.join(path_dir, event), label))


## TODO: Baiyu 
class EEGDatasetForPCA(Dataset):
    pass



if __name__ == "__main__":
    dataset = EEGDataset('../../data', 'full')
    dataset = EEGDataset('../../data', 'subject', subjects=[1])
    dataset = EEGDataset('../../data', 'task', subjects=[1], tasks=[3,4])
    print(len(dataset))