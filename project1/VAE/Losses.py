import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDCE(nn.Module):
    def __init__(self):
        super(KLDCE, self).__init__()

    def forward(self,org,rec,log_var,mean):
        sizes=org.size(0)
        MSE=F.mse_loss(rec,org,reduction="sum")
        KL=-0.5*torch.sum(1+log_var-mean.pow(2)-log_var.exp())
        loss=KL
        return loss
        
