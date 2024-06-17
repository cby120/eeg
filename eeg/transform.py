import torch
from torch import nn
from typing import List
# from scipy.signal import cwt, ricker
from pywt import cwt


class CWT_(nn.Module):
    """
    perform wavelet transform to EEG data [[N,] T, C]
    :param widths: sliding window widths for wavelet transformation
        ~ T or 1/freq
    :param sample: batched or unbatched sample data in tensor [[N,] T, C]
    :return: [[N,] T, C, F]
    """
    def __init__(self, widths: List[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.widths = widths
    
    def forward(self, sample: torch.Tensor):
        x, y = sample
        batched = True
        if x.dim() == 2:  # unbatched data
            x = x.unsqueeze(0)
            batched = False
        assert x.dim() == 3, "input must be 2d or 3d(batched) tensor"
        # x: [N, T, C] => [N, C, T]
        x = x.transpose(1, 2)
        x_ = \
        torch.stack(tuple(  # batch
            torch.stack(tuple(  # channel
                torch.tensor(
                    # cwt(ch, ricker, self.widths)  # scipy [F, T]
                    cwt(ch, self.widths, "morl")[0]  # pywt [F, T] (,T)
                ) for ch in samp
            ))  # [C, F, T]
            for samp in x
        ))  # [N, C, F, T]
        x_ = x_.permute(0, 3, 1, 2)  # => [N, T, C, F]
        x_ = x_ if batched else x_.squeeze(0)
        return x_, y  # [N, T, C, F]


class CWT(nn.Module):
    """
    perform wavelet transform to EEG data [[N,] T, C]
    :param widths: sliding window widths for wavelet transformation
        ~ T or 1/freq
    :param sample: batched or unbatched sample data in tensor [[N,] T, C]
    :return: [[N,] T, C, F]
    """
    def __init__(self, widths: List[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.widths = widths
    
    def forward(self, sample: torch.Tensor):
        x, y = sample
        batched = True
        if x.dim() == 2:  # unbatched data
            x = x.unsqueeze(0)
            batched = False
        assert x.dim() == 3, "input must be 2d or 3d(batched) tensor"
        x = x.double().numpy()
        # x: [N, T, C] => [N, C, T]
        # x = x.transpose(1, 2)
        
        x_, freq = cwt(x, self.widths, "morl", axis=1)  # [F, N, T, C] 
        # print(x_.shape)
        x_ = torch.tensor(x_).permute(1, 2, 3, 0)  # =>[N, T, C, F]
        x_ = x_ if batched else x_.squeeze(0)
        return x_, y  # [N, T, C, F]


if __name__ == "__main__":
    from data_utils import EEGDataset
    d = EEGDataset("../data")
    # print(d[1000][0].shape)
    f = CWT([1, 2, 3, 4, 5])
    # print(f(d[1000])[0].shape)
    from torch.utils.data import DataLoader
    # dl = DataLoader(d, 3, collate_fn=)
    # y = f(next(iter(dl)))
    y = f((torch.stack((d[10][0], d[10][0])), torch.stack((d[10][1], d[10][1]))))
    print(y[0].shape)
    print(y[1])
