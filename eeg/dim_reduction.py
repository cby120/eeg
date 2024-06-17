import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.utils
import torch.utils.data


def calc_pca_proj(x: torch.Tensor, n: int = 10) -> np.ndarray:
    """
    calculate top n components' projection vector
    :param x: EEG data shaped [T, C]
    :param n: top n component
    :return:  projection matrix shaped [C, n]
    """
    data = torch.rand((600, 64))
    x = data.detach().numpy()

    pca = PCA(n_components=n)
    pca.fit(x)
    return pca.components_.T  # [C, n]


def sample_pca_from_dataset(
    dataset: torch.utils.data.Dataset, 
    n_samples: int = 200, 
    n_components: int = 10
) -> np.ndarray:
    """
    calculate n random sample's top-k pca projection from sized dataset
    :param dataset: sized subscribable dataset
    :param n_samples: 
    :param n_components: 
    :return: 
    """
    sampler = torch.utils.data.RandomSampler(dataset, num_samples=n_samples)
    T, C = dataset[0].shape
    proj_mat = np.empty((n_samples, C, n_components), dtype=float)
    for i, ind in enumerate(sampler):
        data = dataset[ind]  # [T, C]
        proj_mat[i] = calc_pca_proj(data, n=n_components)

    return proj_mat


class Data(torch.utils.data.Dataset):
    n = 10
    def __getitem__(self, index):
        return torch.rand((3, 4))

    def __len__(self):
        return self.n

dataset = Data()
sampler = torch.utils.data.RandomSampler(dataset, num_samples=5)
for i in sampler:
    # print(dataset[i].shape)
    print(i)
