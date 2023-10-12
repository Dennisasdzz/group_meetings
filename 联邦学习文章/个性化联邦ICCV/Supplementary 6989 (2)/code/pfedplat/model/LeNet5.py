
import pfedplat as fp
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(fp.Model):

    def __init__(self, device, *args, **kwargs):
        super(LeNet5, self).__init__(device)

        self.input_require_shape = [3, -1, -1]
        self.target_require_shape = []

        self.ignore_head = False

    def generate_net(self, input_data_shape, target_class_num, *args, **kwargs):
        self.name = 'CNN_of_cifar10_tutorial'
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.predictor = nn.Linear(84, target_class_num)
        self.create_Loc_reshape_list()

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.decoder(x)
        if not self.ignore_head:
            x = self.predictor(x)
        return x
