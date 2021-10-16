from typing import List, Tuple
import copy
from custom_types import IStateDict
import torch


def split_state_dict(
    state_dict: IStateDict, bias: bool
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """ return sigma and psi respectively """
    return get_sigma_from_state_dict(state_dict, bias), get_psi_from_state_dict(state_dict, bias)


def get_psi_from_state_dict(
    state_dict: IStateDict, bias: bool
) -> List[torch.Tensor]:
    return [tensor for name, tensor in state_dict.items() if 'psi' in name or (
        ('bias' in name) if bias else False
    )]


def get_sigma_from_state_dict(state_dict: IStateDict, bias: bool) -> List[torch.Tensor]:
    return [tensor for name, tensor in state_dict.items() if 'sigma' in name or (
        ('bias' in name) if bias else False
    )]


def flatten_weight(weights: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.flatten() for t in weights])


def replace_psi_in_state_dict(state_dict: IStateDict, psi: List[torch.Tensor]) -> IStateDict:
    new_state_dict = copy.deepcopy(state_dict)

    idx = 0
    for key in state_dict:
        if 'psi' in key:
            new_state_dict[key] = psi[idx]
            idx += 1

    assert idx == len(psi)

    return new_state_dict


if __name__ == '__main__':
    from base_model import Backbone
    from datasets import generate_random_dataloaders

    model = Backbone(generate_random_dataloaders('cifar10'))
    sd = model.state_dict()
    psi = model.get_psi_tensors()

    replace_psi_in_state_dict(sd, psi)
