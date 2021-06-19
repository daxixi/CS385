import os
import utils
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision import transforms as T
import torchvision.datasets as dset

from torch.autograd import Variable
#from model import NetworkCIFAR as Network
import MLP
import RNN

def save_img(grad, path):
    imgs=[]
    for i in range(100):
        img = grad[i].detach().cpu().permute(1, 2, 0).numpy()
        img -= img.min()
        img /= img.max()
        img = np.uint8(255*img)
        imgs.append(img)

    all_img=imgs[0]
    for j in range(1,10):
        all_img=np.concatenate((all_img,imgs[j]),axis=1)
    for i in range(1,10):
        row_img=imgs[i*10]
        for j in range(1,10):
            row_img=np.concatenate((row_img,imgs[i*10+j]),axis=1)
        all_img=np.concatenate((all_img,row_img),axis=0)
        
    cv2.imwrite(f"{path}", all_img)
    print(f"Save {path} complete")

class GBP(nn.Module):
    def __init__(self,model):
        super(GBP, self).__init__()
        self.bone = model
        self.bone.train()
        self.criterion=nn.CrossEntropyLoss()

        self.set_backprop()

    def set_backprop(self):
        def relu_backward_hook(module, grad_out, grad_in):
            modified_grad_out = nn.functional.relu(grad_out[0])
            return (modified_grad_out, )

        for idx, item in enumerate(self.bone.modules()):
            if isinstance(item, nn.ReLU):
                item.register_backward_hook(relu_backward_hook)


    def generate_gradient(self, input, target):
        input.requires_grad = True
        model_output = self.bone(input)
        self.bone.zero_grad()

        loss=self.criterion(model_output,target)
        
        # Backward through network
        loss.backward()

        # Return the gradient
        return input.grad

    def forward(self, input, target):
        return self.generate_gradient(input, target)

def get_map(img,row,column):

    return all_img

def main():
    model = RNN.get_model(4)
    model_path='eval-ResLSTM-20210614-184229/weights.pt'
    model.load_state_dict(torch.load(model_path))
    model = GBP(model)
    model = model.cuda()
    
    train_transform, valid_transform = utils._data_transforms_svhn()
    valid_data = dset.SVHN('../data', split='test', download=True, transform=valid_transform)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=64, shuffle=False, pin_memory=True, num_workers=0)

    pics={}  
    for step, (input, target) in enumerate(valid_queue):
        input=input.cuda()
        target=target.cuda()
        grad = model(input,target)
        for index,t in enumerate(target):
            if not t.item() in pics:
                pics[t.item()]=[(grad[index],input[index])]
            else:
                if len(pics[t.item()])==10:
                    continue
                else:
                    pics[t.item()].append((grad[index],input[index]))
        flag=0
        for i in range(10):
            if len(pics[i])<10:
                flag=1
                break
        if flag==0:
            break

    grad=[]
    input=[]
    for i in range(10):
        for j in range(10):
            grad.append(pics[i][j][0])
            input.append(pics[i][j][1])
    
    save_img(grad, "grad.png")
    save_img(input, "org.png")
main()
