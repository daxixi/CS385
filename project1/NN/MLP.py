import torch.nn as nn

class Net(nn.Module):
    def __init__(self,num_classes=10,start_channel=3):
        super(Net, self).__init__()
        self.linears=nn.Sequential(nn.Linear(3072,1000),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Linear(1000,1000),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.3),
                                   nn.Linear(1000,1000),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.4),
                                   nn.Linear(1000,1000),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(1000,10))
        
    def forward(self,x):
        x=x.view(x.size(0),-1)
        x=self.linears(x)
        return x
        
def get_model():
    model=Net() 
    return model
