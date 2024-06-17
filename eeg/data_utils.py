import os
import torch
import random
import numpy as np
from typing import List
from torch.utils.data import Dataset, RandomSampler, Subset, DataLoader

N_TASKS = 14
N_SUBJECTS = 109


class EEGDatasetHYB(Dataset):
    task_num = 14
    p = 0.9  # partition ratio of train and test set from full dataset

    def __init__(self, data_dir: str, load_attr: str, train: bool = True, **kwargs):
        """
        :param data_dir: _description_
        :param load_attr: in ("full", "subject", "task")
        :param train: _description_, defaults to True
        """
        self._data_dir = os.path.abspath(data_dir) if not os.path.isabs(data_dir) else data_dir
        self._load_attr = f'_load_{load_attr}'
        self._train = train
        self._meta_data = []

        # load data based on different attribute
        if hasattr(self, self._load_attr):
            load_func = getattr(self, self._load_attr)
            load_func(**kwargs)
        else:
            raise ValueError(f"Attribute '{load_attr}' not found in EEGDataset.")

        # random.shuffle(self._meta_data)
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


class EEGDataset(Dataset):
    p = 0.9  # partition ratio of train and test set from full dataset

    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        subjects: List[int] = None,
        tasks: List[int] = None,
        shuffle: bool = True,
    ):
        """_summary_
        :param data_dir: 
        :param train: 
        :param subjects: [1, 109] white list filter of subjects
        :param tasks: [1, 14] white list filter of tasks
        :data shape: [T, C]
        """
        self._data_dir = os.path.abspath(data_dir) if not os.path.isabs(data_dir) else data_dir
        self._train = train
        self._meta_data = []

        # load data based on different attribute

        self.load_func(subjects, tasks)

        if shuffle:
            random.shuffle(self._meta_data)
        
        if self._train:
            self._meta_data = self._meta_data[:int(self.p * len(self._meta_data))]
        else:
            self._meta_data = self._meta_data[int(self.p * len(self._meta_data)):]

    def __len__(self):
        return len(self._meta_data)

    def __getitem__(self, index: int):
        data_path, label = self._meta_data[index]
        data = np.load(data_path)["data"]
        # print(list(data.keys()))
        return torch.Tensor(data), torch.scalar_tensor(label)

    def load_func(self, subjects: List[int] = None, tasks: List[int] = None):
        if subjects is None:
            subjects = os.listdir(self._data_dir)
        else:
            assert all(_ in range(1, N_SUBJECTS + 1) for _ in subjects), f"subject id must be in [1, {N_SUBJECTS}]"
            subjects = [f"{_:03d}" for _ in subjects]
        if tasks is None:
            tasks = [f"{_:02d}" for _ in range(1, N_TASKS + 1)]
        else:
            assert all(_ in range(1, N_TASKS + 1) for _ in tasks), f"task id must be in [1, {N_TASKS}]"
            tasks = [f"{_:02d}" for _ in tasks]
        for subject in subjects:
            subj_pth = os.path.join(self._data_dir, subject)
            for task in tasks:
                for event in os.listdir(os.path.join(subj_pth, task)):
                    label = int(event.split('_')[0])
                    self._meta_data.append((os.path.join(self._data_dir, subject, task, event), label))


def get_random_subset(dataset: Dataset, n_samples: int):
    sample_ind = list(RandomSampler(dataset, num_samples=n_samples))
    # print(sample_ind)
    return Subset(dataset, sample_ind)


if __name__ == "__main__":
    # dataset = EEGDatasetHYB('../data', 'full')
    # dataset = EEGDatasetHYB('../../data', 'subject', subjects=[1])
    # dataset = EEGDatasetHYB('../../data', 'task', subjects=[1], tasks=[3,4])
    dataset = EEGDataset("../data", subjects=[1, 2, 3], tasks=[2, 3, 4])
    print(len(dataset))
