from datasets.normal_dataset import NormalInputDataset
from datasets.outliers_dataset import OutliersDataset


def generate_training_datasets(normal_sample_size, outliers_sample_size):
    return NormalInputDataset(normal_sample_size, mean=-50.0), OutliersDataset(outliers_sample_size)