import torch

import numpy as np
import torch.nn as nn
from masks.models.VAE import uVAE
import time
from glob import glob
import pdb
import argparse
torch.manual_seed(42)
np.random.seed(42)
from pydicom import dcmread
from skimage.transform import resize
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from skimage.exposure import equalize_hist as equalize
from skimage.io import imread,imsave
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from masks.utils.postProcess import postProcess
from masks.utils.tools import dice,binary_accuracy
from torchvision.utils import save_image
import os
import cv2
plt.gray()

def load(f):
  # options
  new_w = 448

  # open
  ds = dcmread(f)
  dcm = ds.pixel_array

  # rescale
  dcm = dcm / dcm.max()
  if ds.PhotometricInterpretation == 'MONOCHROME1':
    dcm = 1 - dcm

  # fix that shit
  # mapping, w, h = get_minimal_transform(dcm * 255)
  # dcm = minimize(dcm, mapping, w, h)

  # equalize
  dcm = equalize(dcm)

  # crop and resize image to 640x512
  new_h = int(dcm.shape[0] / (dcm.shape[1] / new_w))
  if new_h > 576:
    new_h = 576
    new_w = int((dcm.shape[1]/(dcm.shape[0]/new_h)))

  img = resize(dcm,(new_h,new_w))
  img = torch.Tensor(img)
  pImg = torch.zeros((640, 512))

  print(p)
  h = (int((576-new_h)/2)) + p
  w = int((448-new_w)/2) + p
  roi = torch.zeros(pImg.shape)
  if w == p:
    pImg[np.abs(h):(h+img.shape[0]),p:-p] = img
    roi[np.abs(h):(h+img.shape[0]),p:-p] = 1.0
  else:
    pImg[p:-p,np.abs(w):(w+img.shape[1])] = img
    roi[p:-p,np.abs(w):(w+img.shape[1])] = 1.0

  imH = dcm.shape[0]
  imW = dcm.shape[1]
  pImg = pImg.unsqueeze(0).unsqueeze(0)
  return pImg, roi, h, w, new_h, new_w, imH, imW

def saveMask(f,img,h,w,new_h,new_w,imH,imgW,no_post=False):

  img = img.data.numpy()
  imgIp = img.copy()

  if w == p:
    img = resize(img[np.abs(h):(h+new_h),p:-p],
          (imH,imW),preserve_range=True)
  else:
    img = resize(img[p:-p,np.abs(w):(w+new_w)],
          (imH,imW),preserve_range=True)
  imsave(f,img_as_ubyte(img>0.5))

  if not no_post:
    imgPost = postProcess(imgIp)
    if w == p:
      imgPost = resize(imgPost[np.abs(h):(h+new_h),p:-p],
              (imH,imW))
    else:
      imgPost = resize(imgPost[p:-p,np.abs(w):(w+new_w)],
              (imH,imW))

    imsave(f.replace('.png','_post.png'),img_as_ubyte(imgPost > 0.5))

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='.', help='Path to input files')
parser.add_argument('--model', type=str, default='saved_models/lungVAE.pt', help='Path to trained model')
parser.add_argument('--hidden', type=int, default=16, help='Hidden units')
parser.add_argument('--latent', type=int, default=8, help='Latent dim')
parser.add_argument('--saveLoc', type=str, default='', help='Path to save predictions')
parser.add_argument('--unet',action='store_true', default=False,help='Use only U-Net.')
parser.add_argument('--dicom',action='store_true', default=False,help='DICOM inputs.')
parser.add_argument('--no_post',action='store_true', default=False,help='Do not post process predictions')
parser.add_argument('--padding', type=int, default=32, help='Zero padding')

args = parser.parse_args()
p = args.padding
print("Loading "+ args.model)
if 'unet' in args.model:
  args.unet = True
  args.hidden = int(1.5 * args.hidden)
else:
  args.unet = False

device = torch.device('cuda:1')
net = uVAE(nhid = args.hidden, nlatent=args.latent, unet=args.unet).to(device)
net.load_state_dict(torch.load(args.model, map_location=device))
t = time.strftime("%Y%m%d_%H_%M")

if args.saveLoc == '':
  save_dir = args.data + 'pred_' + t + '/'
else:
  save_dir = args.saveLoc+'pred_' + t + '/'

if not os.path.exists(save_dir):
  os.mkdir(save_dir)

nParam = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Model "+args.model.split('/')[-1]+" Number of parameters:%d"%(nParam))

files = os.listdir(args.data)
files = list(map(lambda x: os.path.join(args.data, x), files))

files = sorted(files)
for fIdx in range(len(files)):
  print('Trying to do something')
  f = files[fIdx]
  fName = f.split('/')[-1]
  img, roi, h, w, new_h, new_w, imH, imW = load(f)
  img = img.to(device)
  _,mask = net(img)
  mask = mask.cpu()
  print(mask.shape)
  mask = torch.sigmoid(mask*roi)
  f = save_dir + fName.replace('.dcm', '_mask.png')

  saveMask(f,mask.squeeze(),h,w,new_h,new_w,imH,imW,args.no_post)
  print("Segmenting %d/%d"%(fIdx,len(files)))
