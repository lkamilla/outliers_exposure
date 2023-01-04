from torch.utils.data import Dataset
from torch.distributions import Normal

class NormalInputDataset(Dataset):

    def __init__(self, sample_size, mean: float = 0.0, variance: float = 1.0, ):
        self.sample_size = (sample_size, 2)
        self.distribution = Normal(mean, variance)
        self.sample = self.distribution.sample(self.sample_size)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        return self.sample[idx]