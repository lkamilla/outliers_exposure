from torch.distributions import Gamma, Beta
from torch.utils.data import Dataset


class OutliersDataset(Dataset):

    def __init__(self, sample_size, concentration=1.0, rate=1.0, transform=None):
        self.transform = transform
        self.sample_size = (sample_size, 2)
        self.distribution = Gamma(concentration, rate)
        self.sample = self.distribution.sample(self.sample_size)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        point = self.sample[idx]
        return point