import os.path
from datetime import datetime

import torch
from torch import nn
from torch.distributions import Beta
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.main import generate_2_dims_training_datasets
from networks.discriminator import discriminator_nn
from networks.generator import generator_nn

device = 'cpu'


def disc_loss(disc_pred, ground_truth_val):
    criterion = nn.MSELoss()
    ground_truth = torch.ones_like(disc_pred) * ground_truth_val
    loss = criterion(disc_pred, ground_truth)
    return loss


def gen_loss(gen_point, ground_truth):
    criterion = nn.MSELoss()
    ground_truth = ground_truth.view(gen_point.shape)
    loss = criterion(gen_point, ground_truth)
    return loss


def create_file():
    filename = f"out_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    file_src = os.path.join("out", filename)
    file_src=filename
    f = open(file_src, 'x')
    return f


def train(epochs, batch_size, lambda_1, lambda_2, alfa, beta, beta_1, beta_2, learning_rate):
    out_file = create_file()
    out_file.write(f"Started training file with epochs: {epochs}, batch_size: {batch_size}, "
                   f"lambda_1: {lambda_1}, lambda_2: {lambda_2}, alfa: {alfa}, beta: {beta}, "
                   f"beta_1: {beta_1}, beta_2: {beta_2}, learning_rate: {learning_rate}\n")
    discriminator_optimizer = torch.optim.Adam(discriminator_nn.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
    generator_optimizer = torch.optim.Adam(generator_nn.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
    beta_distribution = Beta(torch.tensor([alfa]), torch.tensor([beta]))
    normal_dataset, outliers_dataset = generate_2_dims_training_datasets(1000, 40)
    normal_trainloader = DataLoader(normal_dataset, shuffle=True, batch_size=batch_size)
    outliers_trainloader = DataLoader(outliers_dataset, batch_size=batch_size, shuffle=True)
    for i in range(epochs):
        total_discriminator_loss = 0.0
        total_generator_loss = 0.0

        for normal_point in tqdm(normal_trainloader):
            normal_point = normal_point.to(device)
            for outlier in outliers_trainloader:
                outlier = outlier.to(device)

                interpolation = beta_distribution.sample((batch_size,))
                interpolation = interpolation.to(device)

                gen_input = torch.cat((normal_point, outlier, interpolation), dim=1)
                gen_input.to(device)

                # find loss and update weights for the discriminator nn

                discriminator_optimizer.zero_grad()
                fake_outlier = generator_nn(gen_input)
                fake_outlier_ = fake_outlier.squeeze()
                # print(f" Fake outlier shape: {fake_outlier_}")
                disc_pred_interp = discriminator_nn(fake_outlier_)
                disc_loss_value = disc_loss(disc_pred_interp, interpolation)

                zero_interp_input = torch.cat((normal_point, outlier, torch.zeros_like(interpolation, device=device)),
                                              dim=1)
                zero_interp_output = generator_nn(zero_interp_input)
                zero_interp_loss = gen_loss(zero_interp_output, normal_point)

                one_interp_input = torch.cat((normal_point, outlier, torch.ones_like(interpolation, device=device)),
                                             dim=1)
                one_interp_output = generator_nn(one_interp_input)
                one_interp_loss = gen_loss(one_interp_output, outlier)

                gen_loss_value = zero_interp_loss * lambda_1 + one_interp_loss * lambda_2

                disc_loss_value.backward()
                discriminator_optimizer.step()

                # find loss and update weights for generator

                generator_optimizer.zero_grad()
                gen_loss_value.backward()
                generator_optimizer.step()

                total_discriminator_loss += disc_loss_value
                total_generator_loss += gen_loss_value

        avg_disc_loss = total_discriminator_loss / (len(normal_trainloader) * len(outliers_trainloader))
        avg_gen_loss = total_generator_loss / (len(normal_trainloader) * len(outliers_trainloader))

        if i % 50 == 0:
            print(f"Epoch: {i + 1} | D: loss {avg_disc_loss} | G: loss {avg_gen_loss}")
            out_file.write(f"Epoch: {i + 1} | D: loss {avg_disc_loss} | G: loss {avg_gen_loss}\n")
    out_file.close()
