import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, dropout):
        """"Constructor of the class"""
        super(LSTMCell, self).__init__()

        self.nlayers = nlayers
        self.dropout = nn.Dropout(p=dropout)

        ih, hh = [], []
        for i in range(nlayers):
            ih.append(nn.Linear(input_size, 4 * hidden_size))
            hh.append(nn.Linear(hidden_size, 4 * hidden_size))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)

    def forward(self, input, hidden):
        """"Defines the forward computation of the LSTMCell"""
        hy, cy = [], []
        for i in range(self.nlayers):
            hx, cx = hidden[0][i], hidden[1][i]
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)

            i_gate = F.sigmoid(i_gate)
            f_gate = F.sigmoid(f_gate)
            c_gate = F.ReLU(c_gate)
            o_gate = F.sigmoid(o_gate)

            ncx = (f_gate * cx) + (i_gate * c_gate)
            nhx = o_gate * F.tanh(ncx)
            cy.append(ncx)
            hy.append(nhx)
            input = self.dropout(nhx)

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)
        return hy, cy

class Net3(nn.Module):
    def __init__(self,num_classes=10,start_channel=3):
        super(Net3, self).__init__()
        self.LSTM = nn.LSTM(32*3,128,num_layers=10)
        self.output = nn.Linear(128,10)
        
    def forward(self,x):
        x=x.view(x.size(0),32,32*3)
        x,_=self.LSTM(x)
        x=x[:,-1,:]
        x=self.output(x)
        return x
    
class Net(nn.Module):
    def __init__(self,num_classes=10,start_channel=3):
        super(Net, self).__init__()
        self.LSTM = nn.LSTM(32*3,128,batch_first=True,num_layers=3,
                            dropout=0,bidirectional=True)
        self.output = nn.Linear(128*2,10)
        
    def forward(self,x):
        x=x.view(x.size(0),32,32*3)
        x,_=self.LSTM(x)
        y=x[:,-1,:]
        x=self.output(y)
        return x,y

class Net4(nn.Module):
    def __init__(self,num_classes=10,start_channel=3):
        super(Net4, self).__init__()
        self.LSTM = nn.LSTM(3072,1024,batch_first=True,num_layers=1)
        self.output = nn.Linear(1024,10)
        
    def forward(self,x):
        x=x.view(x.size(0),1,3072)
        x=x.repeat(1,16,1)
        x,_=self.LSTM(x)
        x=x[:,-1,:]
        x=self.output(x)
        return x
    
class Net2(nn.Module):
    def __init__(self,num_classes=10,start_channel=3):
        super(Net2, self).__init__()
        self.LSTM = nn.RNN(32*3,128,batch_first=True,num_layers=3,bidirectional=True)
        self.output = nn.Linear(128*2,10)
        
    def forward(self,x):
        x=x.view(x.size(0),32,32*3)
        x,_=self.LSTM(x)
        x=x[:,-1,:]
        x=self.output(x)
        return x

def get_model(types):
    if types==0:
        model=Net() 
    elif types==1:
        model=Net2()
    elif types==2:
        model=Net3()
    else:
        model=Net4()
    return model
