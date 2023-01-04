from torch import nn


class Discriminator(nn.Module):

    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

    def forward(self, point):
        out = self.layers(point)
        return out


discriminator_nn = Discriminator()