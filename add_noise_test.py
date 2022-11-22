print('d')
import  cv2
import numpy as np
from torch.autograd import Variable
import torch

def variable_to_cv2_image(varim):
	r"""Converts a torch.autograd.Variable to an OpenCV image

	Args:
		varim: a torch.autograd.Variable
	"""
	nchannels = varim.size()[1]
	if nchannels == 1:
		res = (varim.data.cpu().numpy()[0, 0, :]*255.).clip(0, 255).astype(np.uint8)
	elif nchannels == 3:
		res = varim.data.cpu().numpy()[0]
		res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
		res = (res*255.).clip(0, 255).astype(np.uint8)
	else:
		raise Exception('Number of color channels not supported')
	return res

sigma = 250
imorig = cv2.imread('69b.jpg')
imorig = cv2.resize(imorig,[720,720])

imorig = (cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
imorig = np.expand_dims(imorig, 0)
# Handle odd sizes
expanded_h = False
expanded_w = False
sh_im = imorig.shape
if sh_im[2] % 2 == 1:
    expanded_h = True
    imorig = np.concatenate((imorig, \
                             imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

if sh_im[3] % 2 == 1:
    expanded_w = True
    imorig = np.concatenate((imorig, \
                             imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
    
    
imorig = np.float32((imorig)/255.)
imorig = torch.Tensor(imorig)



noise = torch.FloatTensor(imorig.size()). \
    normal_(mean=0, std=sigma/255.)
imnoisy = imorig + noise
dtype = torch.cuda.FloatTensor




with torch.no_grad():  # PyTorch v0.4.0
    imorig, imnoisy = Variable(imorig.type(dtype)), \
                      Variable(imnoisy.type(dtype))
    nsigma = Variable(
        torch.FloatTensor([sigma/255.]).type(dtype))







if expanded_h:
    imorig = imorig[:, :, :-1, :]
    imnoisy = imnoisy[:, :, :-1, :]

if expanded_w:
    imorig = imorig[:, :, :, :-1]
    imnoisy = imnoisy[:, :, :, :-1]




noisyimg = variable_to_cv2_image(imnoisy)
noisyimg = cv2.resize(noisyimg,[180,180])
cv2.imwrite("noisy.png", noisyimg)
