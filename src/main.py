from train import train

import click

################################################################################
# Settings
################################################################################
import torch



@click.command()


@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for the network training. Default=0.001')
@click.option('--n_epochs', type=int, default=20, help='Number of epochs to train.')
@click.option('--batch_size', type=int, default=10, help='Batch size for mini-batch training.')
@click.option('--lambda_one', type=float, default=0.5, help='Lambda 1 in loss function')
@click.option('--lambda_two', type=float, default=0.5, help='Lambda 2 in loss function')


def main(lr, n_epochs, batch_size, lambda_one, lambda_two):
    torch.manual_seed(42)
    train(n_epochs, batch_size, lambda_one, lambda_two, 0.5, 0.5, 0.5, 0.99, lr)


if __name__ == '__main__':
    main()