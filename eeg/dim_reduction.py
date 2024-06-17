import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm

# print("hello world!")


def calc_pca(x: torch.Tensor, n: int = 10) -> PCA:
    """
    calculate top n components' projection vector
    :param x: EEG data shaped [T, C]
    :param n: top n component
    :return:  pca obj, .components_ [n, C]
    """
    data = torch.rand((600, 64))
    x = data.detach().numpy()

    pca = PCA(n_components=n)
    pca.fit(x)
    return pca


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
    # T, C = dataset[0][0].shape
    # proj_mat = np.empty((n_samples, C, n_components), dtype=float)
    pca_list = list()
    for i, ind in tqdm(enumerate(sampler)):
        data, _ = dataset[ind]  # [T, C]
        pca = calc_pca(data, n=n_components)
        # pca
        # proj_mat[i] = pca.components_.T
        pca_list.append(pca)
    # residual = np.mean(list(pca.explained_variance_ratio_ for pca in pca_list))
    # print(residual)
    return pca_list


if __name__ == "__main__":
    from data_utils import EEGDataset
    dataset = EEGDataset(data_dir="../data", shuffle=False, train=True)
    # sampler = torch.utils.data.RandomSampler(dataset, num_samples=5)
    # for i in sampler:
    #     # print(dataset[i].shape)
    #     print(i)
    # print(dataset._meta_data[0])
    # print(dataset[0])
    n = 5
    sample_pca_from_dataset(dataset, 1000, n)
    sample_pca_from_dataset(dataset, 1000, n)
    sample_pca_from_dataset(dataset, 1000, n)
