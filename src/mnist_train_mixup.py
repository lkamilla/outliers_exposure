import os.path
import uuid
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Beta, Uniform
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.main import generate_mnist_training_datasets, generate_mnist_training_datasets_with_exclude
from networks.mnist_discriminator import discriminator_nn
from networks.mnist_generator import generator_nn
from torchvision.utils import make_grid, save_image
from matplotlib import pyplot as plt

from datasets.mixed_dataset import CustomMixedMNISTDataset


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


def disc_loss(disc_pred, ground_truth_val):
    criterion = nn.MSELoss()
    ground_truth = ground_truth_val.view(disc_pred.shape)
    loss = criterion(disc_pred, ground_truth)
    return loss


def gen_loss(gen_point, ground_truth):
    criterion = nn.MSELoss(reduction="sum")
    ground_truth = ground_truth.view(gen_point.shape)
    loss = criterion(gen_point.float(), ground_truth.float())
    return loss


def create_file():
    filename = f"out_mnist_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    file_src = os.path.join(".", "out", filename)
    f = open(filename, 'x')
    return f


def create_dir(normal_dataset_size: int, outliers_dataset_size: int, lambda_one: float, lambda_two: float,
               interpolation_size: int, lr: float, epochs: int):
    path = os.path.join("out", str(uuid.uuid4()))
    Path(path).mkdir(exist_ok=True, parents=True)
    params_filename = os.path.join(path, "params.txt")
    with open(params_filename, "x") as params_file:
        params_file.write(f"Size of normal dataset: {normal_dataset_size}\n"
                          f"Number of outliers: {outliers_dataset_size}\n"
                          f"Interpolation sample size: {interpolation_size}\n"
                          f"lambda_1: {lambda_one}\n"
                          f"lambda_2: {lambda_two}\n"
                          f"Learning rate: {lr}\n"
                          f"Epochs: {epochs}")
    return path




def train(epochs, batch_size, lambda_1, lambda_2, alpha, beta_1, beta_2, learning_rate, interpolation_sample_size):
    normal_dataset_size = batch_size * 2
    outliers_dataset_size = 1
    path = create_dir(normal_dataset_size, outliers_dataset_size, lambda_1, lambda_2, interpolation_sample_size, learning_rate, epochs)
    dataset = CustomMixedMNISTDataset(normal_dataset_size, outliers_dataset_size, 1, 2)
    discriminator_nn.apply(weights_init)
    generator_nn.apply(weights_init)
    discriminator_optimizer = torch.optim.Adam(discriminator_nn.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
    generator_optimizer = torch.optim.Adam(generator_nn.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
    uniform_distribution = Uniform(0, 1)
    interpolations = uniform_distribution.sample((interpolation_sample_size, 1))
    trainloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    avg_disc_loss = 0
    avg_gen_loss = 0
    for i in range(1, epochs + 1):
        total_discriminator_loss = 0.0
        total_generator_loss = 0.0
        for data in trainloader:
            disc_batch_loss = 0.0
            for interpolation in interpolations:
                interpolation_batch = torch.ones(batch_size, 1) * interpolation
                generated_outlier = generator_nn(data, interpolation_batch)
                regressed_interpolation = discriminator_nn(generated_outlier)
                disc_ineterp_loss = disc_loss(regressed_interpolation, interpolation_batch)
                disc_batch_loss += disc_ineterp_loss
            disc_batch_loss /= interpolation_sample_size
            disc_batch_loss.backward()
            discriminator_optimizer.step()
            generated_total_outlier = generator_nn(data, torch.ones(batch_size, 1))
            generated_total_normal = generator_nn(data, torch.zeros(batch_size, 1))
            gen_loss_outlier_value = gen_loss(generated_total_outlier, data[:, 1, :, :])
            gen_loss_normal_value = gen_loss(generated_total_normal, data[:, 0, :, :])
            gen_loss_value = gen_loss_normal_value * lambda_1 + gen_loss_outlier_value * lambda_2
            gen_loss_value.backward()
            generator_optimizer.step()
            total_generator_loss += gen_loss_value
            total_discriminator_loss += disc_batch_loss

        avg_disc_loss = total_discriminator_loss / len(trainloader)
        avg_gen_loss = total_generator_loss / (len(trainloader))

        if i % 10 == 0:
            print(f"Epoch: {i} | D: loss {avg_disc_loss} | G: loss {avg_gen_loss}")
            save_images(dataset, os.path.join(path, f"epoch_{i}.pdf"), avg_disc_loss, avg_gen_loss)

        save_images(dataset, os.path.join(path, f"epoch_{i}.pdf"), avg_disc_loss, avg_gen_loss)

def save_images(dataset, src, d_error, g_error):
    generator_nn.eval()
    discriminator_nn.eval()
    interpolations = torch.arange(0, 1.1, step=0.1)
    generated_outliers = []
    random_indices = torch.randint(0, len(dataset), (10, ))
    data = [dataset[idx] for idx in random_indices]
    with torch.no_grad():
        for pair in data:
            generated_outliers.append((pair[0, :, :], "normal"))
            for interpolation in interpolations:
                generated_outlier = generator_nn(pair, interpolation.unsqueeze(0))
                prediction = discriminator_nn(generated_outlier)
                generated_outliers.append(
                    (generated_outlier, f"Gamma: {interpolation: .3f}, Pred: {float(prediction): .3f}"))
            generated_outliers.append((pair[1, :, :], "outlier"))

    _, axs = plt.subplots(5, 13, figsize=(28, 28))
    axs = axs.flatten()
    for (img, title), ax in zip(generated_outliers, axs):
        img_arr = img.detach().numpy().reshape(28, 28)
        ax.imshow(img_arr)
        ax.set_title(title, fontsize=6)
    plt.suptitle(f"D error: {d_error}, G error {g_error}")
    plt.savefig(src)
    generator_nn.train()
    discriminator_nn.train()


