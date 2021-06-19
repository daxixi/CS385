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

import MobileNet
import NiN
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE

matplotlib.use('Agg')

def main():
    #model=MobileNet.get_model()
    model=NiN.get_model()
    model_path='eval-NiN-20210531-182047/weights.pt'
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    
    train_transform, valid_transform = utils._data_transforms_svhn()
    valid_data = dset.SVHN('../data', split='test', download=True, transform=valid_transform)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=128, shuffle=False, pin_memory=True, num_workers=0)

    features=None
    labels=None
    pics={}  
    for step, (input, target) in enumerate(valid_queue):
        for index,t in enumerate(target):
            if not t.item() in pics:
                pics[t.item()]=[input[index]]
            else:
                if len(pics[t.item()])==1:
                    continue
                else:
                    pics[t.item()].append(input[index])
        flag=0
        for i in range(10):
            p=pics[i]
            if len(p)<1:
                flag=1
                break
        if flag==0:
            break
    '''
    cams=[ScoreCAM, AblationCAM]
    names=['ScoreCAM','AblationCAM']
    for CAM,name in zip(cams,names):
        visuals=[]
        for i in range(10):
            #target_layer=model.layers1[-1]
            #target_layer=model.features[-11]
            target_layer=model.block1[-1]
            cam=CAM(model=model,target_layer=target_layer,use_cuda=True)
            grayscale_cam=cam(input_tensor=torch.tensor([pics[i][0].numpy().tolist()]))
            grayscale_cam = grayscale_cam[0, :]
            img=pics[i][0].detach().cpu().permute(1, 2, 0).numpy()
            img -= img.min()
            img /= img.max()
            visualization=show_cam_on_image(img,grayscale_cam)
            visuals.append(visualization)

        all_img=visuals[0]
        for j in range(1,10):
            all_img=np.concatenate((all_img,visuals[j]),axis=1)
            
        cv2.imwrite("vgg19block1_"+name+".png", all_img)
        
    '''
    cams=[GradCAM]#, GradCAMPlusPlus, XGradCAM, EigenCAM]#ScoreCAM, AblationCAM, 
    names=['gradCAM']#,  'GradCAMPlusPlus',  'XGradCAM', 'EigenCAM']#'ScoreCAM','AblationCAM',
    for CAM,name in zip(cams,names):
        #if not name=='EigenCAM':
        #    continue
        visuals=[]
        for i in range(10):
            target_layer=model.classifier[-29]
            #target_layer=model.layers[-3]
            cam=CAM(model=model,target_layer=target_layer,use_cuda=True)
            grayscale_cam=cam(input_tensor=torch.tensor([pics[i][0].numpy().tolist()]))
            grayscale_cam = grayscale_cam[0, :]
            img=pics[i][0].detach().cpu().permute(1, 2, 0).numpy()
            img -= img.min()
            img /= img.max()
            visualization=show_cam_on_image(img,grayscale_cam)
            visuals.append(visualization)

        all_img=visuals[0]
        for j in range(1,10):
            all_img=np.concatenate((all_img,visuals[j]),axis=1)
            
        cv2.imwrite("vgg19block1_"+name+".png", all_img)
    

    
main()
