import PIL.Image
import torch
from torch.autograd import Variable
import torch.nn as nn
import skimage
import skimage.io as io
from torchvision import transforms
import numpy as np
import scipy.io as scio
import argparse
import os
import PIL

from modelNetM import EncoderNet, DecoderNet, ClassNet, EPELoss


pr = argparse.ArgumentParser()
pr.add_argument('--modelspath', type=str)
pr.add_argument('--imgspath', type=str)
pr.add_argument('--flowpath', type=str)
args = pr.parse_args()

models_path = args.modelspath
imgs_path = args.imgspath
flow_path = args.flowpath


#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose([
    transforms.Resize((256, 256)), # your resolution
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model_en = EncoderNet([1,1,1,1,2])
model_de = DecoderNet([1,1,1,1,2])
model_class = ClassNet()

if torch.cuda.is_available():
    model_en = model_en.cuda()
    model_de = model_de.cuda()
    model_class = model_class.cuda()


model_en.load_state_dict(torch.load(os.path.join(models_path, 'model_en_last.pkl')), strict=True)
model_de.load_state_dict(torch.load(os.path.join(models_path, 'model_de_last.pkl')), strict=True)
model_class.load_state_dict(torch.load(os.path.join(models_path, 'model_class_last.pkl')), strict=True)

model_en.eval()
model_de.eval()
model_class.eval()  

testImgPath = imgs_path
saveFlowPath = flow_path

correct = 0
#for index, types in enumerate(['barrel','pincushion','rotation','shear','projective','wave']):
#for index, types in enumerate(['barrel', 'rotation', 'shear', 'wave']):
for index, types in enumerate(['distorted']):
    for k in range(6, 7):

        imgPath = '%s%s%s%s%s%s' % (testImgPath, '\\', types, '_', str(k).zfill(6), '.png')
        disimgs = PIL.Image.open(imgPath).convert('RGB')
        #img_npy = np.asarray(disimgs.resize((128, 128))) # your resolution
        img_npy = np.asarray(disimgs)
        disimgs = transform(disimgs)
        
        use_GPU = torch.cuda.is_available()
        if use_GPU:
            disimgs = disimgs.cuda()
        
        disimgs = disimgs.view(1,3,256,256) # your resolution
        disimgs = Variable(disimgs)
        
        middle = model_en(disimgs)
        flow_output = model_de(middle)
        clas = model_class(middle)
        
        _, predicted = torch.max(clas.data, 1)
        if predicted.cpu().numpy()[0] == index:
            correct += 1

        u = flow_output.data.cpu().numpy()[0][0]
        v = flow_output.data.cpu().numpy()[0][1]

        from resample.resampling import rectification
        res_img, re_mask = rectification(img_npy, flow_output.data.cpu().numpy()[0])
        img_out = PIL.Image.fromarray(res_img)
        #img_out = img_out.resize((128, 128)) # your resolution
        img_out.save(f'output/result_{types}.png')

        saveMatPath =  '%s%s%s%s%s%s' % (saveFlowPath, '/', types, '_', str(k).zfill(6), '.mat')
        scio.savemat(saveMatPath, {'u': u,'v': v})
