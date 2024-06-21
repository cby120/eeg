import os
import torch
import random
import numpy as np
from typing import List
from torch.utils.data import Dataset, RandomSampler, Subset, DataLoader
import pickle
from typing import List, Union, Callable, Any, Iterable
N_TASKS = 14
N_SUBJECTS = 109
DURATION = 6.25e-3


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

#how to split train and val
class EEGDataset(Dataset):
    p = 0.9  # partition ratio of train and test set from full dataset
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        transforms: Union[Callable, Iterable[Callable]] = None,
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
        if transforms is not None:
            if isinstance(transforms, Callable):
                self.transforms = [transforms]
            elif isinstance(transforms, Iterable):
                self.transforms = list(transforms)
            else:
                raise ValueError("transforms must be Callable or Iterable of Callable")
        else:
            self.transforms = None
        pkl_name = '2_t_600_remove_some_people.pkl'
        # load data based on different attribute
        if os.path.exists(pkl_name):
            with open(pkl_name, 'rb') as f:
                self._meta_data = pickle.load(f)
        else:
            self._meta_data = []
            self.load_func(subjects, tasks)
            with open(pkl_name, 'wb') as f:
                pickle.dump(self._meta_data, f)

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
        data = torch.tensor(np.load(data_path)["data"])
        label = torch.scalar_tensor(label, dtype=torch.int)
        # print(list(data.keys()))
        if self.transforms is not None:
            for t in self.transforms:
                data = t(data)
        return data, label

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
                    #print(label)
                    if label !=0 and label !=3 and label !=4:
                        meta_path = os.path.join(self._data_dir, subject, task, event)
                        data = np.load(meta_path)["data"]
                        t,c = data.shape 
                        if t >= 600: #and label!= 0:
                            self._meta_data.append((meta_path, label))
                            #print(self._meta_data)

                    
def get_random_subset(dataset: Dataset, n_samples: int):
    """
    Random Sampled dataset subset
    :param dataset: 
    :param n_samples: 
    :return: Subset randomly sampled from dataset
    """
    sample_ind = list(RandomSampler(dataset, num_samples=n_samples))
    # print(sample_ind)
    return Subset(dataset, sample_ind)


#两个collect——fn
def trim_last(batch):
    """_summary_
    collate_fn for dataloader
    :param batch: _description_
    :type batch: _type_
    :return: _description_
    :rtype: _type_
    """
    x_list, y_list = tuple(zip(*batch))
    x_list: List[torch.Tensor]  # [T, C]
    len_list = np.array(list(_.size(0) for _ in x_list))
    min_len = 100 #len_list.min()#
    x_list_ = [_[:min_len] for _ in x_list]
    return torch.stack(x_list_), torch.stack(y_list)
    
def trim_symm(batch):
    """
    collate_fn for dataloader
    :param batch: List[Tuple[x, y], ...]
    :return: batched data
    """
    x_list, y_list = tuple(zip(*batch))
    x_list: List[torch.Tensor]  # [T, C]
    # print(x_list[0].shape)
    len_list = np.array(list(_.size(0) for _ in x_list))
    # print(len_list)
    min_len = 600 #len_list.min()
    dl = len_list - min_len
    ltil = dl // 2  # list_trim_index_left
    ltir = len_list - dl + ltil
    x_list_ = [_[til: tir] for _, til, tir in zip(x_list, ltil, ltir)]
    return torch.stack(x_list_), torch.stack(y_list)

# def trim_symm_3d(batch):
#     """
#     collate_fn for dataloader
#     :param batch: List[Tuple[x, y], ...]
#     :return: batched data
#     """
#     x_list, y_list = tuple(zip(*batch))
#     x_list: List[torch.Tensor]  # [T, C, F]
#     # print(x_list[0].shape)
#     len_list = np.array(list(_.size(0) for _ in x_list))
#     # print(len_list)
#     min_len = 600 #len_list.min()
#     dl = len_list - min_len
#     ltil = dl // 2  # list_trim_index_left
#     ltir = len_list - dl + ltil
#     x_list_ = [_[til: tir] for _, til, tir in zip(x_list, ltil, ltir)]
#     return torch.stack(x_list_), torch.stack(y_list)



if __name__ == "__main__":
    # dataset = EEGDatasetHYB('../data', 'full')
    # dataset = EEGDatasetHYB('../../data', 'subject', subjects=[1])
    # dataset = EEGDatasetHYB('../../data', 'task', subjects=[1], tasks=[3,4])
    # dataset = EEGDataset("./data", tasks=[3,4,5,6,7,8,9,10,11,12,13,14],subjects= [x for x in range(1, 110) if x not in [88, 90, 92, 100, 104, 106]])
    #print(len(dataset)) 
    #35612 35414
    train_dataset = EEGDataset(data_dir="./data", tasks=[3,4,5,6,7,8,9,10,11,12,13,14],subjects= [x for x in range(1, 110) if x not in [88, 90, 92, 100, 104, 106]],shuffle=False, train=True)

    # from braindecode.datasets import MOABBDataset
    # from braindecode.preprocessing.windowers import create_windows_from_events
    # from braindecode.preprocessing.preprocess import (preprocess, Preprocessor)
    # # # dl = DataLoader(dataset, 10, collate_fn=trim_symm)
    # dl = DataLoader(dataset, 10, collate_fn=trim_last)
    # print(next(iter(dl)))
