
import pfedplat as fp
import torch


class FedSGD(fp.Algorithm):
    """
    该FedSGD实际叫Gossip-SGD
    """

    def __init__(self,
                 name='FedSGD',
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
        epochs = 1

        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, online_client_num,
                         metric_list, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, params)

    def run(self):

        batch_num = min(self.send_require_attr('training_batch_num'))
        if batch_num <= 0:
            raise RuntimeError('error in FedMGDAP: batch_num must > 0')

        training_nums = torch.Tensor(self.send_require_attr(
            'local_training_number')).float().to(self.device)

        weights = training_nums / torch.sum(training_nums)
        while not self.terminated(update_count=True):

            self.send_train_order(self.epochs)

            m_locals, l_locals = self.send_require_client_model()

            g_locals = []
            old_models = self.model.span_model_params_to_vec()
            for idx, client in enumerate(m_locals):
                grad = old_models - m_locals[idx].span_model_params_to_vec()
                g_locals.append(grad)
            g_locals = torch.stack(g_locals)

            d = weights @ g_locals

            for i, p in enumerate(self.model.parameters()):
                p.grad = d[self.model.Loc_reshape_list[i]]
            self.optimizer.step()

            self.current_training_num += self.epochs * batch_num
