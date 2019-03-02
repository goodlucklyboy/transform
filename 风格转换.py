from __future__ import print_function

import torch as t
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

import copy
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
imsize = 512 if use_cuda else 128
loader = transforms.Compose([transforms.Scale(imsize),transforms.ToTensor()])
style_img_path = 'style.jpg'
content_img_path = 'content.jpg'
def image_loader(image_name):
	image = Image.open(image_name)
	image = Variable(loader(image))
	image = image.unsqueeze(0)
	return image


style_img = image_loader(style_img_path).type(dtype)
content_img = image_loader(content_img_path).type(dtype)

assert style_img.size() == content_img.size()

# ***********************************************************************
#Display images

unloader = transforms.ToPILImage()
plt.ion()


def figure_imshow(tensor,title=None):
	image = tensor.cpu()
	image = image.view(3,imsize,imsize)
	image = unloader(image)
	plt.imshow(image)
	if title is not None:
		plt.title(title)
	plt.pause(1)
	
	
plt.figure()
figure_imshow(style_img.data, title='Style Image')

plt.figure()
figure_imshow(content_img.data, title='Content Image')

def GramMatrix(input):
	a, b, c, d = input.size()
	features = input.view(a*b, c*d)
	G = torch.mm(features, features.t())
	return G.div(a*b*c*d)

	
	
"""      定义ContentLoss类         """


class ContentLoss(nn.Module):
	def __init__(self,target,weight):
		super(ContentLoss,self).__init__()
		self.target = target.detach()*weight
		self.weight = weight
		self.criterion = nn.MSELoss()
		self.loss = 0
		self.output = 0
		
	def forward(self,input):
		self.loss = self.criterion(input * self.weight, self.target)
		self.output = input
		return self.output
	
	
"""      定义StyleLoss类       """


class StyleLoss(nn.Module):#这个类的定义目前有问题，且看后面如何使用这个类来返回查看
	
	def __init__(self, target, weight):
		super(StyleLoss,self).__init__()
		self.target = target.detach()*weight
		self.weight = weight
		self.gram = GramMatrix()
		self.criterion = nn.MSELoss()
	
	def forward(self, input):
		self.output = input.clone()
		self.gram_matrix= self.gram(input)
		self.gram_matrix.mul_(self.weight)
		loss = self.criterion(self.gram_matrix,self.target)
		return self.output
	
	def backward(self,retain_graph=True):
		self.loss.backward(retain_graph=retain_graph)
		return self.loss
	

	
"""Load the neural network"""


cnn = models.vgg19(pretrained=True).features

if use_cuda:
	cnn = cnn.cuda()
	
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(net, style_img, content_img, style_weight=1000,
                               content_weight=1, content_layers=content_layers_default,
                               style_layers=style_layers_default):
	cnn = copy.deepcopy(net)
	content_losses = []
	style_losses = []
	model = nn.Sequential()
	if use_cuda:
		model = model.cuda()
	i = 1
	for layer in list(cnn):
		if isinstance(layer,nn.Conv2d):
			name = "conv"+str(i)
			model.add_module(name,layer)
		
			if name in content_layers:
				target = model(content_img).clone()
				content_loss = ContentLoss(target,content_weight)
				model.add_module("content_loss"+str(i),content_loss)
				content_losses.append(content_loss)
			
			if name in style_layers:
				target_feature = model(style_img).clone()
				target_feature_gram = GramMatrix(target_feature)
				style_loss = StyleLoss(target_feature_gram,style_weight)
				model.add_module("style_loss"+str(i), style_loss)
				style_losses.append(style_loss)
		
		if isinstance(layer,nn.ReLU):
			name = "relu"+str(i)
			model.add_module(name, layer)
			
			if name in content_layers:
				target = model(content_img).clone()
				content_loss = ContentLoss(target,content_weight)
				model.add_module("content_loss"+str(i),content_loss)
				content_losses.append(content_loss)
				
			if name in style_layers:
				target_feature = model(style_img).clone()
				target_feature_gram = GramMatrix(target_feature)
				style_loss = StyleLoss(target_feature_gram,style_weight)
				model.add_module("style_loss"+str(i), style_loss)
				style_losses.append(style_loss)
			
			i += 1
		if isinstance(layer,nn.MaxPool2d):
			name = "pool_"+str(i)
			model.add_module(name,layer)
	
	return model,style_losses,content_losses


"""input image"""

input_img = content_img.clone()
figure_imshow(input_img.data,title='input image')


def get_input_param_optimizer(input_img):
	input_param = nn.Parameter(input_img.data)
	optimizer = optim.LBFGS([input_param])
	return input_param, optimizer

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=1000,
                       style_weight= 1000, content_weight=1):

	print('Buiding the style transfer model..')
	model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img,
	                                                                 content_img,style_weight, content_weight)
	input_param, optimizer = get_input_param_optimizer(input_img)
	print('Optimizing..')
	run = [0]
	while run[0]<= num_steps:
		def closure():
			input_param.data.clamp_(0, 1)
			
			optimizer.zero_grad()
			model(input_param)
			style_score = 0
			content_score = 0
			
			for s1 in style_losses:
				style_score += s1.backward()
			for c1 in content_losses:
				content_score += c1.backward()
				
			run[0] += 1
			if run[0]%50 == 0:
				print("run{}:".format(run))
				print('Style Loss:{:4f} '
				      'ContentLoss:{:4f}'.format(style_score,content_score))
			
			return style_score+content_score
		optimizer.step(closure)
		
		
	input_param.data.clamp_(0,1)
	return input_param.data


output = run_style_transfer(cnn,content_img,style_img,input_img)
plt.figure()
figure_imshow(output,title='OUTPUT iamge')

plt.ioff()
plt.show()

	
				
				
				
				
		
