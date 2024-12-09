import torch
from torch import nn
import sys
sys.path.insert(1, './baseline/EpHod')
from ephod import models

class pHPredictionModel(nn.Module):
    def __init__(self, pretrained=True):
        super(pHPredictionModel, self).__init__()
        self.ephod_model = models.EpHodModel().train()
        
        self._set_parameter_requires_grad()
        
        if not pretrained:
            self._initialize_trainable_params()
        
        self._set_batchnorm_to_eval()

    def _set_parameter_requires_grad(self):
        for name, param in self.ephod_model.named_parameters():
            param.requires_grad = 'rlat_model' in name

    def _initialize_trainable_params(self):
        for param in self.ephod_model.parameters():
            if param.requires_grad:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)

    def _set_batchnorm_to_eval(self):
        for module in self.ephod_model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
    
    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"trainable parameters: {trainable_params}")
        print(f"total parameters: {total_params}")
        print(f"trainable parameters ratio: {trainable_params / total_params:.2%}")

    def forward(self, accs, sequences):
        ephod_preds, _, _ = self.ephod_model.batch_predict(accs, sequences)
        return ephod_preds

