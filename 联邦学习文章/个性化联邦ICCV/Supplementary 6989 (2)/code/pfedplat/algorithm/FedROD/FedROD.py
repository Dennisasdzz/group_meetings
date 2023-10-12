
import pfedplat as fp
import numpy as np
import torch
import copy
import torch.nn.functional as F
from torch.autograd import Variable
import time


class FedROD(fp.Algorithm):
    def __init__(self,
                 name='FedROD',
                 data_loader=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 online_client_num=None,
                 metric_list=None,
                 max_comm_round=0,
                 max_training_num=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 write_log=True,
                 dishonest=None,
                 params=None,
                 *args,
                 **kwargs):

        if client_num is None and client_list is not None:
            client_num = len(client_list)
        elif client_num is not None and client_list is None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [Client(i, copy.deepcopy(model), device, train_setting,
                                  metric_list, data_loader.target_class_num) for i in range(client_num)]
            data_loader.allocate(client_list)
        elif client_num is None and client_list is None:
            raise RuntimeError(
                'Both of client_num and client_list cannot be None or not None.')

        for idx, client in enumerate(client_list):
            msg = {'command': 'cal_sample_per_class'}
            client.receive_fedrod_message(msg)

        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, params)
        self.exist_per_model = True

    def run(self):

        training_nums = self.send_require_attr('local_training_number')

        batch_num = np.mean(self.send_require_attr('training_batch_num'))
        while not self.terminated(update_count=True):
            com_time_start = time.time()

            self.send_train_order(self.epochs)

            w_locals, _ = self.send_require_training_result()
            com_time_end = time.time()
            cal_time_start = time.time()

            self.aggregate(w_locals, training_nums)

            self.current_training_num += self.epochs * batch_num * 2

            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start

    def aggregate(self, w_locals, training_nums):
        training_num = sum(training_nums)
        averaged_params = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number = training_nums[i]
                local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        self.model.load_state_dict(averaged_params)


class Client(fp.Client):
    def __init__(self,
                 id=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 class_num=10,
                 *args,
                 **kwargs):
        super().__init__(id, model, device, train_setting, metric_list, *args, **kwargs)

        self.per_model = copy.deepcopy(model)
        self.per_optimizer = train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.per_model.parameters()), lr=self.lr)
        self.per_optimizer.defaults = copy.deepcopy(
            train_setting['optimizer'].defaults)
        self.class_num = class_num

        self.sample_per_class = torch.zeros(self.class_num)

    def receive_fedrod_message(self, msg):
        if msg['command'] == 'cal_sample_per_class':
            self.cal_sample_per_class()

    def cal_sample_per_class(self):

        for x, y in self.local_training_data:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        self.sample_per_class = self.sample_per_class / \
            torch.sum(self.sample_per_class)

    def model_forward(self, model, x):

        model.ignore_head = True
        z = model(x)

        out1 = model.predictor(z)

        out2 = self.per_model.predictor(z)
        output = out1 + out2
        model.ignore_head = False
        return output

    def train(self, epochs):

        global_model_params = list(self.model.parameters())
        for idx, p in enumerate(self.per_model.parameters()):
            if idx < len(global_model_params) - 1:
                p.data = global_model_params[idx].data.clone()
                p.requires_grad = False

        self.model.train()
        for e in range(epochs):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)

                out = self.model(batch_x)
                loss = self.balanced_softmax_loss(
                    batch_y, out, self.sample_per_class)

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

                out = self.model_forward(self.model, batch_x)
                loss = self.criterion(out, batch_y)

                self.per_model.zero_grad()
                loss.backward()
                self.per_optimizer.step()

        self.model_loss = float(loss)

    def balanced_softmax_loss(self, labels, logits, sample_per_class, reduction="mean"):
        spc = sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        loss = F.cross_entropy(
            input=logits, target=labels, reduction=reduction)
        return loss
