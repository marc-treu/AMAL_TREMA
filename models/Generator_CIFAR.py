import torch


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encode = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),

            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),

            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),

            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),

            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),

            torch.nn.Conv2d(32, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),

            torch.nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.encode(x)


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decode = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3,
                            stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),

            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),

            torch.nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),

            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),

            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),

            torch.nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),

            torch.nn.Conv2d(16, 3, 1, padding=1)
        )

    def forward(self, x):
        return torch.tanh(self.decode(x))


