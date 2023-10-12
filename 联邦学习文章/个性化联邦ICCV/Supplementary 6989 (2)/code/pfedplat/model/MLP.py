
import pfedplat as fp
import torch
import torch.nn.functional as F


class MLP(fp.Model):
    def __init__(self, device, *args, **kwargs):
        super(MLP, self).__init__(device)
        self.name = 'MLP'

        self.input_require_shape = [-1]
        self.target_require_shape = []

        self.ignore_head = False

    def generate_net(self, input_data_shape, target_class_num, *args, **kwargs):
        input_dim = input_data_shape[0]
        self.fc1 = torch.nn.Linear(input_dim, 200)
        self.fc2 = torch.nn.Linear(200, 200)
        self.predictor = torch.nn.Linear(200, target_class_num)
        self.create_Loc_reshape_list()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        if not self.ignore_head:
            x = self.predictor(x)
        return x
