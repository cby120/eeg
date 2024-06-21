import torch
from torch import nn
from torch.utils import data
import numpy as np
from typing import List
# from scipy.signal import cwt, ricker
from pywt import cwt
from sklearn.decomposition import PCA, FastICA
from data_utils import get_random_subset


# class CWT_(nn.Module):
#     """
#     perform wavelet transform to EEG data [[N,] T, C]
#     :param widths: sliding window widths for wavelet transformation
#         ~ T or 1/freq
#     :param sample: batched or unbatched sample data in tensor [[N,] T, C]
#     :return: [[N,] T, C, F]
#     """
#     def __init__(self, widths: List[int], *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.widths = widths
    
#     def forward(self, sample: torch.Tensor):
#         x, y = sample
#         batched = True
#         if x.dim() == 2:  # unbatched data
#             x = x.unsqueeze(0)
#             batched = False
#         assert x.dim() == 3, "input must be 2d or 3d(batched) tensor"
#         # x: [N, T, C] => [N, C, T]
#         x = x.transpose(1, 2)
#         x_ = \
#         torch.stack(tuple(  # batch
#             torch.stack(tuple(  # channel
#                 torch.tensor(
#                     # cwt(ch, ricker, self.widths)  # scipy [F, T]
#                     cwt(ch, self.widths, "morl")[0]  # pywt [F, T] (,T)
#                 ) for ch in samp
#             ))  # [C, F, T]
#             for samp in x
#         ))  # [N, C, F, T]
#         x_ = x_.permute(0, 3, 1, 2)  # => [N, T, C, F]
#         x_ = x_ if batched else x_.squeeze(0)
#         return x_, y  # [N, T, C, F]


class CWT(nn.Module):
    """
    perform wavelet transform to EEG data [[N,] T, C]
    :param widths: sliding window widths for wavelet transformation
        ~ T or 1/freq
    :param sample: batched or unbatched sample data in tensor [[N,] T, C]
    :return: [[N,] T, C, F]
    """
    def __init__(self, widths: List[int] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        channal = 5
        if widths is not None:
            self.widths = widths
        else:
            self.widths = np.round((np.exp(np.linspace(0, 2.5, channal)) - 1) * 30, 0).astype(int) + 1
        
    def forward(self, x: torch.Tensor):
        batched = True
        if x.dim() == 2:  # unbatched data
            x = x.unsqueeze(0)
            batched = False
        assert x.dim() == 3, "input must be 2d or 3d(batched) tensor"
        x = x.double().numpy()
        
        x_, freq = cwt(x, self.widths, "morl", axis=1)  # [F, N, T, C] 
        # print(x_.shape)
        x_ = torch.tensor(x_).permute(1, 2, 3, 0)  # =>[N, T, C, F]
        x_ = x_ if batched else x_.squeeze(0)
        return x_  # [[N,] T, C, F]


class PersudoFT(nn.Module):
    """
    perform wavelet transform to EEG data [[N,] T, C]
    pass to dataset(transforms=...) directly or in Iterable
    :param widths: sliding window widths for wavelet transformation
        ~ T or 1/freq
    :param sample: batched or unbatched sample data in tensor [[N,] T, C]
    :return: [[N,] T, F]
    """
    def __init__(self, widths: List[int] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if widths is not None:
            self.widths = widths
        else:
            self.widths = np.round((np.exp(np.linspace(0, 2.5, 26)) - 1) * 30, 0).astype(int) + 1
        
    def forward(self, x: torch.Tensor):
        batched = True
        if x.dim() == 2:  # unbatched data
            x = x.unsqueeze(0)
            batched = False
        assert x.dim() == 3, "input must be 2d or 3d(batched) tensor"
        x = x.double().numpy()
        
        x_, freq = cwt(x, self.widths, "morl", axis=1)  # [F, N, T, C] 
        # print(x_.shape)
        x_ = torch.tensor(x_).permute(1, 2, 0, 3)  # =>[N, T, F, C]
        x_ = x_.std(dim=1)  # => [N, F, C]
        x_ = x_ if batched else x_.squeeze(0)
        return x_  # [[N,] F, C]


class ICATransform(nn.Module):
    """
    Fit and apply ICA decomposition and dim reduction
    pass to dataset(transforms=...) directly or in Iterable
    :param dataset: EEG dataset, Subscrible[torch.Tensor[T, C]]
    :param n_components: recommend range [15, 25]
    :param x: torch.Tensor [T, C]
    :return:
        in forward or __call__
        [T, C] => [T, n_components]
    """
    def __init__(self, dataset: data.Dataset, n_samples: int = 1000, n_components = 20) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.n_components = n_components
        self.ica = FastICA(n_components=n_components)
        self.prepare(dataset)
    
    def prepare(self, dataset: data.Dataset):
        subset = get_random_subset(dataset, self.n_samples)  # Iterable[Tensor[T, C]]
        concat = torch.cat(list(x for x, y in subset), dim=0)  # Tensor[nT, C]
        self.ica.fit(X=concat.cpu().double().numpy())
    
    def forward(self, x: torch.Tensor):
        return torch.tensor(self.ica.transform(x.cpu().double().numpy()))
    

class PCATransform(nn.Module):
    """
    Fit and apply PCA decomposition and dim reduction
    pass to dataset(transforms=...) directly or in Iterable
    :param dataset: EEG dataset, Subscrible[torch.Tensor[T, C]]
    :param n_components: recommend range [10, 20]
    :param x: torch.Tensor [T, C]
    :return:
        in forward or __call__
        [T, C] => [T, n_components]
    """
    def __init__(self, dataset: data.Dataset, n_samples: int = 1000, n_components = 20) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.prepare(dataset)
    
    def prepare(self, dataset: data.Dataset):
        subset = get_random_subset(dataset, self.n_samples)  # Iterable[Tensor[T, C]]
        concat = torch.cat(list(x for x, y in subset), dim=0)  # Tensor[nT, C]
        self.pca.fit(X=concat.cpu().double().numpy())
    
    def forward(self, x: torch.Tensor):
        return torch.tensor(self.pca.transform(x.cpu().double().numpy()))


if __name__ == "__main__":
    from data_utils import EEGDataset
    d = EEGDataset("../data")
    # print(d[1000][0].shape)
    f = CWT([1, 2, 3, 4, 5])
    # print(f(d[1000])[0].shape)
    from torch.utils.data import DataLoader, Data
    import torch
    DataLoader()
    # dl = DataLoader(d, 3, collate_fn=)
    # y = f(next(iter(dl)))
    y = f((torch.stack((d[10][0], d[10][0])), torch.stack((d[10][1], d[10][1]))))
    print(y[0].shape)
    print(y[1])
