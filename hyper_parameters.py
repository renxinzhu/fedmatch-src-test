class ParametersBase:
    """ Just a base class allowing class attribute iteration for HyperParameter and GlobalSetting """

    def items(self):
        return [(a, getattr(self, a)) for a in dir(self) if (
            not a.startswith('__') and a not in (
                'items', 'pretty_print', 'to_string_log')
        )]

    def pretty_print(self):
        for key, value in self.items():
            print(f'{key}: {value}')

    def to_string_log(self):
        string_log = []
        for key, value in self.items():
            string_log.append(f'{key}: {value}')
        return '\n'.join(string_log)


class HyperParameters(ParametersBase):
    local_epochs = 1
    confidence_threshold = 0.75
    lr = 1e-3
    lambda_s = 10
    lambda_iccs = 1e-2
    lambda_l1 = 1e-4
    lambda_l2 = 10


class GlobalSetting(ParametersBase):
    num_clients = 100
    R = 0.05
    rounds = 300
    batch_size_s = 10
    batch_size_u = 100
    dataset_name = 'cifar10'
    label_ratio = 0.05
    iid = True
    h_interval = 10
    num_helpers = 2
    lr_decay_rate = 0.99


hyper_parameters = HyperParameters()
setting = GlobalSetting()
