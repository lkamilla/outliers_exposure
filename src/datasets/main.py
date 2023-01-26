from datasets.normal_dataset import NormalInputDataset
from datasets.outliers_dataset import OutliersDataset
from datasets.mnist_dataset import CustomMNISTDataset

def generate_2_dims_training_datasets(normal_sample_size, outliers_sample_size):
    return NormalInputDataset(normal_sample_size, mean=-50.0), OutliersDataset(outliers_sample_size)


def generate_mnist_training_datasets(normal_sample_size, outliers_sample_size):
    return CustomMNISTDataset(normal_sample_size, 1), CustomMNISTDataset(outliers_sample_size, 8)