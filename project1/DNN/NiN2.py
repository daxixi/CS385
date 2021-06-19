import torch.nn as nn

class lastmodule(nn.Module):
    def __init__(self,num_classes):
        super(lastmodule,self).__init__()
        self.layer=nn.Sequential(nn.Conv2d(192,  num_classes, kernel_size=1, stride=1, padding=0),
                      nn.ReLU(inplace=True),
                      nn.AvgPool2d(kernel_size=8, stride=1, padding=0))
        self.classes=num_classes
        
    def forward(self,x):
        x=self.layer(x)
        x=x.view(x.size(0),self.classes)
        return x

class Net(nn.Module):
    def __init__(self,num_classes=10,start_channel=3):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2, bias=False),
                nn.Tanh(),
                nn.BatchNorm2d(192),
                nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Tanh(),
                nn.BatchNorm2d(160),
                nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Tanh(),
                nn.BatchNorm2d(96),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2, bias=False),
                nn.Tanh(),
                nn.BatchNorm2d(192),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Tanh(),
                nn.BatchNorm2d(192),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Tanh(),
                nn.BatchNorm2d(192),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Tanh(),
                nn.BatchNorm2d(192),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Tanh(),
                nn.BatchNorm2d(192),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.Tanh(),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

                )
    def forward(self,x):
        x=self.classifier(x)
        logits=x.view(x.size(0),10)
        return logits
        
def get_model():
    model=Net()   
    return model
