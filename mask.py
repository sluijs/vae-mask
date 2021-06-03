import os

import numpy as np
import torch

from skimage.transform import resize
from masks.utils.postProcess import postProcess

from masks.models.VAE import uVAE

def preprocess(img):
  """Preprocess the image according to the original implementation"""

  # crop and resize to 640x512
  new_w = 448
  new_h = int(img.shape[0] / (img.shape[1] / new_w))
  if new_h > 576:
    new_h = 576
    new_w = int(img.shape[1] / (img.shape[0] / new_h))
    if new_w % 2 == 1:
      new_w += 1

  img = resize(img, (new_h, new_w))
  img = torch.tensor(img)
  canvas = torch.zeros((640, 512))

  padding = 32
  h = int((576 - new_h) / 2) + padding
  w = int((448 - new_w) / 2) + padding
  roi = torch.zeros_like(canvas)

  if w == padding:
    canvas[np.abs(h):(h + img.shape[0]), padding:-padding] = img
    roi[np.abs(h):(h + img.shape[0]), padding:-padding] = 1.0
  else:
    canvas[padding:-padding, np.abs(w):(w + img.shape[1])] = img
    roi[padding:-padding, np.abs(w):(w + img.shape[1])] = 1.0

  img_h, img_w = img.shape
  canvas = canvas.unsqueeze(0).unsqueeze(0)
  return canvas, roi, h, w, new_h, new_w, img_h, img_w

def generate(img, device='cuda:1'):
  """Generate a lung mask for an image"""

  # generate the mask
  device = torch.device(device)
  net = uVAE(nhid=16, nlatent=8, unet=False).to(device)
  path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models/lungVAE.pt')
  net.load_state_dict(torch.load(path, map_location=device))

  # preprocess the image
  img, roi, h, w, new_h, new_w, img_h, img_w = preprocess(img)
  img = img.to(device)

  _, mask = net(img)
  mask = mask.cpu()
  mask = torch.sigmoid(mask * roi)

  # post-process
  padding = 32
  mask = postProcess(mask.squeeze())
  if w == padding:
    mask = resize(mask[np.abs(h):(h + new_h), padding:-padding], (img_h, img_w))
  else:
    mask = resize(mask[padding:-padding, np.abs(w):(w + new_w)], (img_h, img_w))

  return mask > 0.5

