from torch import nn


class Generator(nn.Module):
    def __init__(self, noise_dim) -> None:
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        self.layers = nn.Sequential(nn.Linear(noise_dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, input_vector):
        output = self.layers(input_vector)
        return output


generator_nn = Generator(5)
