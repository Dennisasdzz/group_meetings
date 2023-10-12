
import pfedplat as fp
import numpy as np
import torch
import copy


class Client:

    def __init__(self,
                 id=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 dishonest=None,
                 *args,
                 **kwargs):
        self.id = id
        if model is not None:
            model = model
        self.model = model
        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        self.train_setting = train_setting
        self.metric_list = metric_list
        self.dishonest = dishonest
        self.local_training_data = None
        self.local_training_number = 0
        self.local_test_data = None
        self.local_test_number = 0
        self.global_test_data = None
        self.global_test_number = 0
        self.training_batch_num = 0
        self.test_batch_num = 0

        self.metric_history = {'training_loss': [],
                               'test_loss': [],
                               'local_test_number': 0,
                               'global_test_loss': [],
                               'global_test_number': 0,
                               'simulated_test_loss': [],
                               'simulated_test_number': 0}
        self.per_model_metric_history = {'test_loss': [],
                                         'global_test_loss': [],
                                         'simulated_test_loss': []}
        for metric in self.metric_list:
            self.metric_history[metric.name] = []
            self.metric_history['global_test_' + metric.name] = []
            self.metric_history['simulated_test_' + metric.name] = []
            self.per_model_metric_history[metric.name] = []
            self.per_model_metric_history['global_test_' + metric.name] = []
            self.per_model_metric_history['simulated_test_' + metric.name] = []
            if metric.name == 'correct':
                self.metric_history['test_accuracy'] = []
                self.metric_history['global_test_accuracy'] = []
                self.metric_history['simulated_test_accuracy'] = []
                self.per_model_metric_history['test_accuracy'] = []
                self.per_model_metric_history['global_test_accuracy'] = []
                self.per_model_metric_history['simulated_test_accuracy'] = []
        self.model_loss = None
        self.info_msg = {}

        self.initial_lr = float(train_setting['optimizer'].defaults['lr'])
        self.lr = self.initial_lr
        self.optimizer = train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.optimizer.defaults = copy.deepcopy(
            train_setting['optimizer'].defaults)
        self.criterion = self.train_setting['criterion'].to(self.device)

        self.old_model = copy.deepcopy(self.model)

        self.per_model = None
        self.old_per_model = copy.deepcopy(self.model)

        self.model_list = []
        self.old_model_list = []

        self.test_per_model = None

    @staticmethod
    def update_learning_rate(optimizer, lr):
        optimizer.defaults['lr'] = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def update_data(self,
                    id,
                    local_training_data,
                    local_training_number,
                    local_test_data,
                    local_test_number,
                    global_test_data,
                    global_test_number,
                    simulated_test_data,
                    simulated_test_number,):
        self.id = id
        self.local_training_data = local_training_data
        self.local_training_number = local_training_number
        self.local_test_data = local_test_data
        self.local_test_number = local_test_number
        self.global_test_data = global_test_data
        self.global_test_number = global_test_number
        self.simulated_test_data = simulated_test_data
        self.simulated_test_number = simulated_test_number
        self.training_batch_num = len(local_training_data)
        self.test_batch_num = len(local_test_data)

    def get_message(self, msg):
        return_msg = {}

        if msg['command'] == 'sync':

            model_weights = msg['w_global']
            self.model.load_state_dict(model_weights)
            self.old_model.load_state_dict(model_weights)
            return return_msg
        if msg['command'] == 'update_learning_rate':
            current_comm_round = msg['current_comm_round']
            self.lr = self.initial_lr * \
                self.train_setting['lr_decay']**current_comm_round
            fp.Algorithm.update_learning_rate(self.optimizer, self.lr)
            return return_msg
        if msg['command'] == 'cal_all_batches_loss':
            self.info_msg['common_loss_of_all_batches'] = self.cal_all_batches_loss(
                self.model)
            return return_msg
        if msg['command'] == 'cal_all_batches_gradient_loss':
            self.cal_all_batches_gradient_loss()
            return return_msg
        if msg['command'] == 'evaluate':
            batch_idx = msg['batch_idx']
            mode = msg['mode']
            self.evaluate(mode, batch_idx)
            return return_msg
        if msg['command'] == 'train':

            epochs = msg['epochs']
            lr = msg['lr']

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.train(epochs)
            return return_msg
        if msg['command'] == 'test':

            self.test()
            return return_msg
        if msg['command'] == 'require_cal_all_batches_loss_result':

            return_loss = self.info_msg['common_loss_of_all_batches']
            return_msg['l_local'] = return_loss
            return return_msg
        if msg['command'] == 'require_all_batches_gradient_loss_result':

            return_grad = self.info_msg['common_gradient_vec_of_all_batches']
            return_loss = self.info_msg['common_loss_of_all_batches']

            if self.dishonest is not None:
                if self.dishonest['scaled_update'] is not None:
                    return_grad *= self.dishonest['scaled_update']
                if self.dishonest['zero_update'] is not None:
                    return_grad *= 0.0
                if self.dishonest['random_update'] is not None:
                    n = len(return_grad)
                    grad = torch.randn(n).float().to(self.device)
                    return_grad = grad
            return_msg['g_local'] = return_grad
            return_msg['l_local'] = return_loss
            return return_msg

        if msg['command'] == 'require_client_model':

            if msg['requires_grad'] == 'True':
                return_model = copy.deepcopy(self.model)
                return_loss = self.model_loss
            else:
                with torch.no_grad():
                    return_model = copy.deepcopy(self.model)
                    return_loss = self.model_loss
            if self.dishonest is not None:
                if self.dishonest['scaled_update'] is not None:
                    return_model = (return_model - self.old_model) * \
                        self.dishonest['scaled_update'] + self.old_model
                if self.dishonest['zero_update'] is not None:
                    return_model = copy.deepcopy(self.old_model)
                if self.dishonest['random_update'] is not None:
                    old_model_params_span = self.old_model.span_model_params_to_vec()
                    n = len(old_model_params_span)
                    updates = torch.randn(n).float().to(self.device)
                    return_model_params = old_model_params_span + updates
                    return_model.reshape_vec_to_model_params(
                        return_model_params)
            return_msg['m_local'] = return_model
            return_msg['l_local'] = return_loss
            return return_msg
        if msg['command'] == 'require_client_per_model':

            if msg['requires_grad'] == 'True':
                return_model = copy.deepcopy(self.per_model)
                return_loss = self.model_loss
            else:
                with torch.no_grad():
                    return_model = copy.deepcopy(self.per_model)
                    return_loss = self.model_loss
            if self.dishonest is not None:
                if self.dishonest['scaled_update'] is not None:
                    return_model = (return_model - self.old_per_model) * \
                        self.dishonest['scaled_update'] + self.old_per_model
                if self.dishonest['zero_update'] is not None:
                    return_model = copy.deepcopy(self.old_per_model)
                if self.dishonest['random_update'] is not None:
                    old_model_params_span = self.old_per_model.span_model_params_to_vec()
                    n = len(old_model_params_span)
                    updates = torch.randn(n).float().to(self.device)
                    return_model_params = old_model_params_span + updates
                    return_model.reshape_vec_to_model_params(
                        return_model_params)
            return_msg['m_local'] = return_model
            return_msg['l_local'] = return_loss
            return return_msg
        if msg['command'] == 'require_training_result':
            return_model = copy.deepcopy(self.model)
            return_loss = self.model_loss

            if self.dishonest is not None:
                if self.dishonest['scaled_update'] is not None:
                    return_model = (return_model - self.old_model) * \
                        self.dishonest['scaled_update'] + self.old_model
                if self.dishonest['zero_update'] is not None:
                    return_model = copy.deepcopy(self.old_model)
                if self.dishonest['random_update'] is not None:
                    old_model_params_span = self.old_model.span_model_params_to_vec()
                    n = len(old_model_params_span)
                    updates = torch.randn(n).float().to(self.device)
                    return_model_params = old_model_params_span + updates
                    return_model.reshape_vec_to_model_params(
                        return_model_params)
            return_msg['w_local'] = return_model.state_dict()
            return_msg['l_local'] = return_loss
            return return_msg
        if msg['command'] == 'require_weight_avg_grad':
            return_model = copy.deepcopy(self.model)
            return_grad = copy.deepcopy(getattr(self, 'avg_grad'))

            if self.dishonest is not None:
                if self.dishonest['scaled_update'] is not None:
                    return_model = (return_model - self.old_model) * \
                        self.dishonest['scaled_update'] + self.old_model
                    return_grad *= self.dishonest['scaled_update']
                if self.dishonest['zero_update'] is not None:
                    return_model = copy.deepcopy(self.old_model)
                    return_grad *= 0.0
                if self.dishonest['random_update'] is not None:
                    model_params_span = return_model.span_model_params_to_vec()
                    old_model_params_span = self.old_model.span_model_params_to_vec()
                    n = len(model_params_span)
                    r = (torch.rand(n) * 2.0 - 1.0).float().to(self.device)
                    r /= torch.norm(r)
                    return_model_params_span = r * \
                        torch.norm(model_params_span -
                                   old_model_params_span) + old_model_params_span
                    for i, p in enumerate(return_model.parameters()):
                        p.data = return_model_params_span[return_model.Loc_reshape_list[i]]
                    return_grad = r * torch.norm(return_grad)
                if self.dishonest['gaussian'] is not None:
                    model_params_span = return_model.span_model_params_to_vec()
                    n = len(model_params_span)
                    weights = torch.randn(n).float().to(self.device)
                    for i, p in enumerate(return_model.parameters()):
                        p.data = weights[return_model.Loc_reshape_list[i]]
                    old_model_params_span = self.old_model.span_model_params_to_vec()
                    grad = old_model_params_span - weights
                    return_grad = grad / \
                        torch.norm(grad) * torch.norm(return_grad)
            return_msg['w_local'] = return_model.state_dict()
            return_msg['avg_g_local'] = return_grad
            return return_msg
        if msg['command'] == 'require_multi_model_weights':
            w_local_list = []
            for idx, model in enumerate(self.model_list):
                return_model = copy.deepcopy(model)

                if self.dishonest is not None:
                    if self.dishonest['scaled_update'] is not None:
                        return_model = (
                            return_model - self.old_model_list[idx]) * self.dishonest['scaled_update'] + self.old_model_list[idx]
                    if self.dishonest['zero_update'] is not None:
                        return_model = copy.deepcopy(self.old_model_list[idx])
                    if self.dishonest['random_update'] is not None:
                        model_params_span = return_model.span_model_params_to_vec()
                        old_model_params_span = self.old_model_list[idx].span_model_params_to_vec(
                        )
                        n = len(model_params_span)
                        r = (torch.rand(n) * 2.0 - 1.0).float().to(self.device)
                        r /= torch.norm(r)
                        return_model_params_span = r * \
                            torch.norm(
                                model_params_span - old_model_params_span) + old_model_params_span
                        for i, p in enumerate(return_model.parameters()):
                            p.data = return_model_params_span[return_model.Loc_reshape_list[i]]
                    if self.dishonest['gaussian'] is not None:
                        model_params_span = return_model.span_model_params_to_vec()
                        n = len(model_params_span)
                        weights = torch.randn(n).float().to(self.device)
                        for i, p in enumerate(return_model.parameters()):
                            p.data = weights[return_model.Loc_reshape_list[i]]
                w_local_list.append(return_model.state_dict())
            return_msg['w_locals'] = w_local_list
            return return_msg

        if msg['command'] == 'require_test_result':
            return_msg['metric_history'] = copy.deepcopy(self.metric_history)
            return_msg['per_model_metric_history'] = copy.deepcopy(
                self.per_model_metric_history)
            return return_msg
        if msg['command'] == 'require_attribute_value':
            attr = msg['attr']
            return_msg['attr'] = getattr(self, attr)
            return return_msg

    def cal_all_batches_loss(self, model):
        model.train()
        total_loss = 0.0
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)

                out = model(batch_x)
                loss = self.criterion(out, batch_y)
                loss = float(loss)
                total_loss += loss * batch_y.shape[0]
            loss = total_loss / self.local_training_number

        return loss

    def cal_per_batches_loss(self, model):
        model.train()

        loss_vec = []
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)

                out = model(batch_x)
                loss = self.criterion(out, batch_y)
                loss_vec.append(loss * batch_y.shape[0])
        loss_vec = torch.Tensor(loss_vec).float().to(self.device)
        return loss_vec

    def update_model(self, model, optimizer, lr, g):
        self.update_learning_rate(optimizer, lr)
        for i, p in enumerate(model.parameters()):
            p.grad = g[model.Loc_reshape_list[i]]
        optimizer.step()

    def cal_all_batches_gradient_loss(self):
        self.model.train()

        grad_mat = []
        total_loss = 0
        weights = []
        for step, (batch_x, batch_y) in enumerate(self.local_training_data):
            batch_x = fp.Model.change_data_device(batch_x, self.device)
            batch_y = fp.Model.change_data_device(batch_y, self.device)
            weights.append(batch_y.shape[0])

            out = self.model(batch_x)
            loss = self.criterion(out, batch_y)
            total_loss += loss * batch_y.shape[0]

            self.model.zero_grad()
            loss.backward()
            grad_vec = self.model.span_model_grad_to_vec()
            grad_mat.append(grad_vec)
        loss = total_loss / self.local_training_number
        weights = torch.Tensor(weights).float().to(self.device)
        weights = weights / torch.sum(weights)

        grad_mat = torch.stack([grad_mat[i] for i in range(len(grad_mat))])

        g = weights @ grad_mat

        self.info_msg['common_gradient_vec_of_all_batches'] = g
        self.info_msg['common_loss_of_all_batches'] = float(loss)
        self.model_loss = float(loss)

    def evaluate(self, mode, batch_idx):

        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise RuntimeError(
                'error in Client: mode can only be train or model')
        batch_x, batch_y = self.local_training_data[batch_idx]
        batch_x = fp.Model.change_data_device(batch_x, self.device)
        batch_y = fp.Model.change_data_device(batch_y, self.device)

        out = self.model(batch_x)
        loss = self.criterion(out, batch_y)

        self.model.zero_grad()
        loss.backward()

        self.model_loss = float(loss)

    def train(self, epochs):
        self.optimizer = self.train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.optimizer.defaults = copy.deepcopy(
            self.train_setting['optimizer'].defaults)

        if epochs <= 0:
            raise RuntimeError('error in Client: epochs must > 0')

        loss = self.cal_all_batches_loss(self.model)

        self.model_loss = float(loss)

        self.model.train()
        for e in range(epochs):
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)

                out = self.model(batch_x)
                loss = self.criterion(out, batch_y)

                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

    def model_forward(self, model, x):
        return model(x)

    def create_metric_dict(self):
        metric_dict = {'test_loss': 0}
        for metric in self.metric_list:
            metric_dict[metric.name] = 0
        return metric_dict

    def test(self):
        self.model.eval()

        if self.test_per_model is None:
            self.test_per_model = self.per_model

        if self.test_per_model is not None:
            self.test_per_model.eval()
        criterion = self.train_setting['criterion'].to(self.device)

        self.metric_history['training_loss'].append(
            float(self.model_loss) if self.model_loss is not None else None)

        metric_dict = self.create_metric_dict()
        global_test_metric_dict = self.create_metric_dict()
        simulated_test_metric_dict = self.create_metric_dict()
        per_model_metric_dict = self.create_metric_dict()
        per_model_global_test_metric_dict = self.create_metric_dict()
        per_model_simulated_test_metric_dict = self.create_metric_dict()

        with torch.no_grad():

            self.metric_history['local_test_number'] = self.local_test_number
            for (batch_x, batch_y) in self.local_test_data:
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)

                out = self.model(batch_x)
                loss = criterion(out, batch_y).item()
                metric_dict['test_loss'] += float(loss) * batch_y.shape[0]
                for metric in self.metric_list:
                    metric_dict[metric.name] += metric.calc(out, batch_y)

                if self.test_per_model is not None:
                    out = self.model_forward(self.test_per_model, batch_x)
                    loss = criterion(out, batch_y).item()
                    per_model_metric_dict['test_loss'] += float(
                        loss) * batch_y.shape[0]
                    for metric in self.metric_list:
                        per_model_metric_dict[metric.name] += metric.calc(
                            out, batch_y)

            self.metric_history['test_loss'].append(
                metric_dict['test_loss'] / self.local_test_number)
            for metric in self.metric_list:
                self.metric_history[metric.name].append(
                    metric_dict[metric.name])
                if metric.name == 'correct':
                    self.metric_history['test_accuracy'].append(
                        100 * metric_dict['correct'] / self.local_test_number)
            if self.test_per_model is not None:
                self.per_model_metric_history['test_loss'].append(
                    metric_dict['test_loss'] / self.local_test_number)
                for metric in self.metric_list:
                    self.per_model_metric_history[metric.name].append(
                        per_model_metric_dict[metric.name])
                    if metric.name == 'correct':
                        self.per_model_metric_history['test_accuracy'].append(
                            100 * per_model_metric_dict['correct'] / self.local_test_number)

            self.metric_history['global_test_number'] = self.global_test_number
            for (batch_x, batch_y) in self.global_test_data:
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)

                out = self.model(batch_x)
                loss = criterion(out, batch_y).item()
                global_test_metric_dict['test_loss'] += float(
                    loss) * batch_y.shape[0]
                for metric in self.metric_list:
                    global_test_metric_dict[metric.name] += metric.calc(
                        out, batch_y)

                if self.test_per_model is not None:
                    out = self.model_forward(self.test_per_model, batch_x)
                    loss = criterion(out, batch_y).item()
                    per_model_global_test_metric_dict['test_loss'] += float(
                        loss) * batch_y.shape[0]
                    for metric in self.metric_list:
                        per_model_global_test_metric_dict[metric.name] += metric.calc(
                            out, batch_y)

            self.metric_history['global_test_loss'].append(
                global_test_metric_dict['test_loss'] / self.global_test_number)
            for metric in self.metric_list:
                self.metric_history['global_test_' +
                                    metric.name].append(global_test_metric_dict[metric.name])
                if metric.name == 'correct':
                    self.metric_history['global_test_accuracy'].append(
                        100 * global_test_metric_dict['correct'] / self.global_test_number)
            if self.test_per_model is not None:
                self.per_model_metric_history['global_test_loss'].append(
                    per_model_global_test_metric_dict['test_loss'] / self.global_test_number)
                for metric in self.metric_list:
                    self.per_model_metric_history['global_test_' + metric.name].append(
                        per_model_global_test_metric_dict[metric.name])
                    if metric.name == 'correct':
                        self.per_model_metric_history['global_test_accuracy'].append(
                            100 * per_model_global_test_metric_dict['correct'] / self.global_test_number)

            self.metric_history['simulated_test_number'] = self.simulated_test_number
            for (batch_x, batch_y) in self.simulated_test_data:
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)

                out = self.model(batch_x)
                loss = criterion(out, batch_y).item()
                simulated_test_metric_dict['test_loss'] += float(
                    loss) * batch_y.shape[0]
                for metric in self.metric_list:
                    simulated_test_metric_dict[metric.name] += metric.calc(
                        out, batch_y)

                if self.test_per_model is not None:
                    out = self.model_forward(self.test_per_model, batch_x)
                    loss = criterion(out, batch_y).item()
                    per_model_simulated_test_metric_dict['test_loss'] += float(
                        loss) * batch_y.shape[0]
                    for metric in self.metric_list:
                        per_model_simulated_test_metric_dict[metric.name] += metric.calc(
                            out, batch_y)

            self.metric_history['simulated_test_loss'].append(
                simulated_test_metric_dict['test_loss'] / self.simulated_test_number)
            for metric in self.metric_list:
                self.metric_history['simulated_test_' +
                                    metric.name].append(simulated_test_metric_dict[metric.name])
                if metric.name == 'correct':
                    self.metric_history['simulated_test_accuracy'].append(
                        100 * simulated_test_metric_dict['correct'] / self.simulated_test_number)
            if self.test_per_model is not None:
                self.per_model_metric_history['simulated_test_loss'].append(
                    per_model_simulated_test_metric_dict['test_loss'] / self.simulated_test_number)
                for metric in self.metric_list:
                    self.per_model_metric_history['simulated_test_' + metric.name].append(
                        per_model_simulated_test_metric_dict[metric.name])
                    if metric.name == 'correct':
                        self.per_model_metric_history['simulated_test_accuracy'].append(
                            100 * per_model_simulated_test_metric_dict['correct'] / self.simulated_test_number)
