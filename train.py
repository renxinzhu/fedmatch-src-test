import gc
import torch
from client import Client
from server import Server
from datasets import generate_dataloaders, generate_test_dataloader
from logger import Logger
from hyper_parameters import hyper_parameters, setting
import pdb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}.')

logger = Logger()

logger.log('image size: 32x32')
logger.log(hyper_parameters.to_string_log())
logger.log(setting.to_string_log())

dataloaders = generate_dataloaders(
    setting.dataset_name,
    setting.num_clients,
    setting.label_ratio,
    setting.iid,
    setting.batch_size_s,
    setting.batch_size_u
)
test_dataloader = generate_test_dataloader(setting.dataset_name)

clients = [Client(dataloader, hyper_parameters) for dataloader in dataloaders]

server = Server()

for r in range(setting.rounds):
    state_dict = server.state_dict()
    idxes = torch.randint(0, setting.num_clients,
                          (int(setting.num_clients * setting.R),))
    logger.log(f'round {r+1}: { idxes }')

    if r % setting.h_interval == 0:
        server.save_client_psis(clients)
        print('tree constructed')

    for idx in idxes:
        client = clients[idx]

        helper_psis = server.get_helpers_by_direct_calculation(
            setting.num_helpers, client.get_psi_tensors())

        client.local_train(state_dict, helper_psis)

    server.aggregate([clients[idx].state_dict() for idx in idxes])

    acc = server.validate(test_dataloader)

    state = f'round {r+1} acc {acc}'
    logger.log(state)

    gc.collect()
