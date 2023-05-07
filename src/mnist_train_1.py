import os.path
from datetime import datetime

import torch
from torch import nn
from torch.distributions import Beta, Uniform
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.main import generate_mnist_training_datasets_with_exclude
from networks.mnist_discriminator import discriminator_nn
from networks.mnist_generator import generator_nn
from matplotlib import pyplot as plt

device = 'cpu'


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


def train(epochs, batch_size, lambda_1, lambda_2, alpha, beta_1, beta_2, learning_rate, interpolation_sample_size):
    out_file = create_file()
    out_file.write(f"Started training file with epochs: {epochs}, batch_size: {batch_size}, "
                   f"lambda_1: {lambda_1}, lambda_2: {lambda_2}, alfa: {alpha} "
                   f"beta_1: {beta_1}, beta_2: {beta_2}, learning_rate: {learning_rate}\n")
    discriminator_nn.apply(weights_init)
    generator_nn.apply(weights_init)
    discriminator_optimizer = torch.optim.Adam(discriminator_nn.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
    generator_optimizer = torch.optim.Adam(generator_nn.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
    uniform_distribution = Uniform(0, 1)
    interpolations = uniform_distribution.sample((interpolation_sample_size, 1))

    normal_dataset, outliers_dataset = generate_mnist_training_datasets_with_exclude(batch_size*1,1)
    normal_trainloader = DataLoader(normal_dataset, shuffle=True, batch_size=batch_size)
    outliers_trainloader = DataLoader(outliers_dataset, batch_size=batch_size, shuffle=True)
    avg_disc_loss = 0
    avg_gen_loss = 0
    filename = f"out_mnist_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    for i in range(1, epochs + 1):
        total_discriminator_loss = 0.0
        total_generator_loss = 0.0

        for normal_data in tqdm(normal_trainloader):

        for normal_data in tqdm(normal_trainloader):
            for outlier in outliers_trainloader:
                generator_input = torch.cat((normal_data, outlier), dim=1)
                disc_batch_loss = 0
                for interpolation in interpolations:
                    interpolation_batch = torch.ones(batch_size, 1) * interpolation
                    generated_outlier = generator_nn(generator_input, interpolation_batch)
                    regressed_interpolation = discriminator_nn(generated_outlier)
                    disc_ineterp_loss = disc_loss(regressed_interpolation, interpolation_batch)
                    disc_batch_loss += disc_ineterp_loss
                disc_batch_loss /= interpolation_sample_size
                disc_batch_loss.backward()
                discriminator_optimizer.step()
                generated_total_outlier = generator_nn(generator_input, torch.ones(batch_size, 1))
                generated_total_normal = generator_nn(generator_input, torch.zeros(batch_size, 1))
                gen_loss_outlier_value = gen_loss(generated_total_outlier, outlier)
                gen_loss_normal_value = gen_loss(generated_total_normal, normal_data)
                gen_loss_value = gen_loss_normal_value * lambda_1 + gen_loss_outlier_value * lambda_2
                gen_loss_value.backward()
                generator_optimizer.step()
                total_generator_loss += gen_loss_value
                total_discriminator_loss += disc_batch_loss

        avg_disc_loss = total_discriminator_loss / (len(normal_trainloader) * len(outliers_trainloader))
        avg_gen_loss = total_generator_loss / (len(normal_trainloader) * len(outliers_trainloader))

        if i % 10 == 0:
        # if True:
            print(f"Epoch: {i} | D: loss {avg_disc_loss} | G: loss {avg_gen_loss}")
            out_file.write(f"Epoch: {i} | D: loss {avg_disc_loss} | G: loss {avg_gen_loss}\n")
            save_images(normal_dataset, outliers_dataset, f"{filename}_epoch_{i}.pdf", avg_disc_loss, avg_gen_loss)
    out_file.close()
    save_images(normal_dataset, outliers_dataset, f"{filename}.pdf", avg_disc_loss, avg_gen_loss)


def save_images(normal_dataset, outliers_dataset, filename, d_error, g_error):
    generator_nn.eval()
    discriminator_nn.eval()
    normal_data = normal_dataset[:5]
    outlier = outliers_dataset[0]
    interpolations = torch.arange(0, 1.1, step=0.1)
    generated_outliers = []
    with torch.no_grad():
        for normal_image in normal_data:
            generator_input = torch.cat((normal_image, outlier), dim=0).unsqueeze(0)
            generated_outliers.append((normal_image, "normal"))
            for interpolation in interpolations:
                generated_outlier = generator_nn(generator_input, interpolation.unsqueeze(0))
                prediction = discriminator_nn(generated_outlier)
                generated_outliers.append((generated_outlier, f"Gamma: {interpolation: .3f}, Pred: {float(prediction): .3f}"))
            generated_outliers.append((outlier, "outlier"))

    _, axs = plt.subplots(5, 13, figsize=(28, 28))
    axs = axs.flatten()
    for (img, title), ax in zip(generated_outliers, axs):
        img_arr = img.detach().numpy().reshape(28, 28)
        ax.imshow(img_arr)
        ax.set_title(title, fontsize=6)
    src = os.path.join('out', filename)
    plt.suptitle(f"D error: {d_error}, G error {g_error}")
    plt.savefig(src)
    generator_nn.train()
    discriminator_nn.train()





