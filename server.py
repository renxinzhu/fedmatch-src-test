from typing import List
from copy import deepcopy
from scipy.spatial import KDTree
import gc

import torch
from base_model import Backbone
from client import Client
from custom_types import IStateDict
from util import flatten_weight


class Server(Backbone):
    def aggregate(self, state_dicts: List[IStateDict]):
        avg_state_dict = deepcopy(state_dicts[0])

        for key in avg_state_dict:
            mean = torch.mean(torch.stack([sd[key] for sd in state_dicts]), 0)
            avg_state_dict[key] = mean

        self.load_state_dict(avg_state_dict)

    def get_helpers_by_kdtree(self, H: int, current_psi: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        """ Get H helpers, returning the clients with the nearest weights """
        toQuery = flatten_weight(current_psi).cpu().numpy()
        _, idxes = self.kdtree.query(toQuery, k=H+1)

        return [self.client_psis[i] for i in idxes[1:]]

    def construct_tree(self, clients: List[Client]):
        self.kdtree = None
        gc.collect()

        # save clients parameters
        self.save_client_psis(clients)

        # construct kdtree
        flatten_psis = [flatten_weight(psi).cpu().numpy()
                        for psi in client_psis]
        self.kdtree = KDTree(flatten_psis)

    def get_helpers_by_direct_calculation(self, H: int, current_psi: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        """ Get H helpers, returning the clients with the nearest weights """

        flattened_psis = [flatten_weight(psi) for psi in self.client_psis]
        flattened_current_psi = flatten_weight(current_psi)

        dist = torch.stack([-torch.sum(torch.abs(psi - flattened_current_psi))
                           for psi in flattened_psis])

        topk_result = torch.topk(dist, H+1, 0)
        idxes = topk_result.indices.tolist()

        return [self.client_psis[i] for i in idxes[1:]]

    def save_client_psis(self, clients: List[Client]):
        client_psis = [list(client.get_psi_tensors()) for client in clients]
        self.client_psis = client_psis
