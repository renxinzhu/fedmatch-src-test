from typing import Dict, cast
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from base_model import Backbone, Model
from hyper_parameters import HyperParameters
from loss import iccs_loss, regularization_loss,src_loss
from custom_types import IStateDict

from typing import List, cast

from util import replace_psi_in_state_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Client(Backbone):
    def __init__(self, dataloader: Dict[str, DataLoader], hyper_parameters: HyperParameters):
        super().__init__()
        self.dataloader = dataloader
        self.hyper_parameters = hyper_parameters

    def local_train(
        self,
        state_dict: IStateDict,
        helper_psis: List[List[torch.Tensor]],
    ):
        self.load_state_dict(state_dict)
        self.helper_psis = helper_psis

        self._train_supervised()
        self._train_unsupervised()

    def _train_supervised(self):
        opt = torch.optim.SGD(self.get_sigma_parameters(
            True), lr=self.hyper_parameters.lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(self.hyper_parameters.local_epochs):
            for __, X, y in self.dataloader['labeled']:
                X = X.to(device)
                y = y.to(device)

                pred,last_feature_map_useless = self.forward(X)
                loss = loss_fn(pred, y) * self.hyper_parameters.lambda_s

                opt.zero_grad()
                loss.backward()
                opt.step()

    def _train_unsupervised(self):
        opt = torch.optim.SGD(self.get_psi_parameters(True), lr=self.hyper_parameters.lr)
        confident_counts = 0

        for _ in range(self.hyper_parameters.local_epochs):
            for noised_X, X, __ in self.dataloader['unlabeled']:
                X = X.to(device)
                noised_X = noised_X.to(device)

                output,_ = self.forward(X)
                pred = F.softmax(output, dim=-1)

                max_values = cast(torch.Tensor, torch.max(pred, dim=1).values)
                confident_idxes = [idx for idx, value in enumerate(max_values.tolist()) if (
                    value > self.hyper_parameters.confidence_threshold
                )]

                confident_counts += len(confident_idxes)

                loss = regularization_loss(
                    self.get_sigma_parameters(False),
                    self.get_psi_parameters(False),
                    self.hyper_parameters.lambda_l1,
                    self.hyper_parameters.lambda_l2,
                )

                if confident_idxes:
                    confident_pred = pred[confident_idxes]
                    confident_noised_X = noised_X[confident_idxes]
                    confident_X = X[confident_idxes]

                    noised_pred,_ = self.forward(confident_noised_X)
                    helper_preds_1,helper_last_feature_maps_1 = self._helper_predictions(confident_X)

                    loss += iccs_loss(confident_pred,
                                      noised_pred, helper_preds_1, self.hyper_parameters.lambda_iccs)
                    ###修改+不确定
                    model = Model()
                    out_useles,local_last_feature_map = model.forward(X)
                    loss += src_loss(local_last_feature_map,helper_last_feature_maps_1,BATCH_SIZE=50)

                opt.zero_grad()
                loss.backward()
                opt.step()

        if confident_counts:
            print('# confident preds: ', confident_counts)

    ###修改
    def _helper_predictions(self, X: torch.Tensor):
        '''make prediction one by one instead of creating all in the RAM to reduce RAM usage'''
        with torch.no_grad():
            helpers_pred = []
            helper_last_feature_maps = []
            for helper_psi in self.helper_psis:
                model = Model()

                state_dict = replace_psi_in_state_dict(
                    self.state_dict(), helper_psi)
                model.load_state_dict(state_dict)

                ###比上src1.0修改了
                pred,helper_last_feature_map = model.forward(X)
                helper_last_feature_maps.append(helper_last_feature_map)
                helpers_pred.append(pred)
            return helpers_pred,helper_last_feature_maps
