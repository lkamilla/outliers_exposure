# from train import train
# from mnist_train import train as train_on_mnist

import click

################################################################################
# Settings
################################################################################
import torch
import time
from mnist_train_mixup import train



@click.command()


@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for the network training. Default=0.001')
@click.option('--n_epochs', type=int, default=100, help='Number of epochs to train.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--lambda_one', type=float, default=0.5, help='Lambda 1 in loss function')
@click.option('--lambda_two', type=float, default=0.5, help='Lambda 2 in loss function')
@click.option('--dataset', type=str, default='default', help='Which dataset train on')
@click.option('--outliers_num', type=int, default=1, help='how many outliers')


def main(lr, n_epochs, batch_size, lambda_one, lambda_two, dataset, outliers_num):
    torch.manual_seed(42)
    # if dataset == 'default':
    #     train(n_epochs, batch_size, lambda_one, lambda_two, 0.5, 0.5, 0.99, lr)
    # else:
    #     train_on_mnist(n_epochs, batch_size, lambda_one, lambda_two, 0.2, 0.5, 0.99, lr, 100)
    #
    #     # train_on_mnist(n_epochs, batch_size, 0.005, 0.001, 0.2, 0.5, 0.99, 0.002, 50)
    #     # train_on_mnist(n_epochs, batch_size, 0.005, 0.001, 0.2, 0.5, 0.99, 0.002, 100)
    #     # train_on_mnist(n_epochs, batch_size, 0.005, 0.005, 0.2, 0.5, 0.99, 0.002, 1000)
    #     # train_on_mnist(n_epochs, batch_size, 0.001, 0.002, 0.2, 0.5, 0.99, 0.002, 1000)
    train(n_epochs, batch_size, outliers_num, 0.05, 0.05, 0.2, 0.5, 0.99, 0.002, 100)


if __name__ == '__main__':
    main()