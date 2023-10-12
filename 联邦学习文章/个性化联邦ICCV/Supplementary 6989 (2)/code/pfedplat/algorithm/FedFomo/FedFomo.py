
import pfedplat as fp
import numpy as np
import copy
import torch


class FedFomo(fp.Algorithm):
    def __init__(self,
                 name='FedFomo',
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
            client_list = [Client(i, copy.deepcopy(model), device, train_setting, metric_list,
                                  data_loader.input_data_shape, data_loader.target_class_num, client_num) for i in range(client_num)]
            data_loader.allocate(client_list)
        elif client_num is None and client_list is None:
            raise RuntimeError(
                'Both of client_num and client_list cannot be None or not None.')

        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, params)
        self.P = torch.diag(torch.ones(self.client_num, device=self.device))
        self.exist_per_model = True

    def run(self):

        batch_num = np.mean(self.send_require_attr('training_batch_num'))
        while not self.terminated():
            m_locals, _ = self.send_require_client_per_model()
            client_id_list = self.send_require_attr('id')
            weight_vector_list = self.send_require_attr('weight_vector')

            self.send_models(m_locals, client_id_list, weight_vector_list)

            self.send_train_order(self.epochs)

            self.aggregate(m_locals)

            self.current_training_num += self.epochs * batch_num

    def aggregate(self, m_locals):
        w_locals = [m_local.state_dict() for m_local in m_locals]
        averaged_params = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_model_params = w_locals[i]
                if i == 0:
                    averaged_params[k] = local_model_params[k] / \
                        self.online_client_num
                else:
                    averaged_params[k] += local_model_params[k] / \
                        self.online_client_num
        self.model.load_state_dict(averaged_params)

    def send_models(self, m_locals, client_id_list, weight_vector_list):

        for idx, client in enumerate(self.online_client_list):
            self.P[client_id_list[idx]] += weight_vector_list[idx]
            choose_indices = torch.topk(
                self.P[client_id_list[idx]][client_id_list], self.online_client_num).indices.tolist()
            model_list = []
            id_list = []
            for i in choose_indices:
                model_list.append(m_locals[i])
                id_list.append(client_id_list[i])
            msg = {'command': 'get_models',
                   'model_list': model_list, 'id_list': id_list}
            client.get_models(msg)

        self.current_comm_round += len(choose_indices)


class Client(fp.Client):
    def __init__(self,
                 id=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 input_data_shape=None,
                 target_class_num=None,
                 client_num=0,
                 *args,
                 **kwargs):
        super().__init__(id, model, device, train_setting, metric_list, *args, **kwargs)
        self.client_num = client_num
        self.weight_vector = torch.zeros(self.client_num, device=device)
        self.client_model_list = []
        self.client_id_list = []

        self.per_model = copy.deepcopy(model)
        self.per_optimizer = train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.per_model.parameters()), lr=self.lr)
        self.per_optimizer.defaults = copy.deepcopy(
            train_setting['optimizer'].defaults)

    def get_models(self, msg):
        if msg['command'] == 'get_models':
            self.client_model_list = msg['model_list']
            self.client_id_list = msg['id_list']

    def train(self, epochs):

        if epochs <= 0:
            raise RuntimeError('error in Client: epochs must > 0')

        self.update_learning_rate(self.per_optimizer, self.lr)
        self.old_per_model = copy.deepcopy(self.per_model)

        self.aggregate()

        self.per_model.train()
        for e in range(epochs):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)

                out = self.per_model(batch_x)
                loss = self.criterion(out, batch_y)

                self.per_model.zero_grad()
                loss.backward()
                self.per_optimizer.step()

        self.model_loss = float(loss)

    def weight_cal(self):
        weight_list = []
        loss = self.cal_all_batches_loss(self.old_model)
        for client_model in self.client_model_list:
            params_diff = []
            for param_n, param_i in zip(client_model.parameters(), self.old_model.parameters()):
                params_diff.append((param_n - param_i).view(-1))
            params_diff = torch.cat(params_diff)
            weight_list.append(float(
                ((loss - self.cal_all_batches_loss(client_model)) / (torch.norm(params_diff) + 1e-6)).item()))

        self.weight_vector = torch.zeros(self.client_num, device=self.device)
        for w, client_id in zip(weight_list, self.client_id_list):
            self.weight_vector[client_id] += w

        for i in range(len(weight_list)):
            if weight_list[i] < 0:
                weight_list[i] = 0.0
        w_sum = np.sum(weight_list)
        weight_list = [w / w_sum for w in weight_list] if w_sum > 0 else []
        return weight_list

    def aggregate(self):

        weight_list = self.weight_cal()
        if len(weight_list) > 0:
            for param in self.per_model.parameters():
                param.data.zero_()
            for weight, client_model in zip(weight_list, self.client_model_list):
                for p, client_model_param in zip(self.per_model.parameters(), client_model.parameters()):
                    p.data += client_model_param.data.clone() * weight
