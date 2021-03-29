import torch
import numpy as np
import pandas as pd
from torch.nn import ReLU, Linear, Module
from torch.nn.init import kaiming_uniform_

class DNN(Module):
	def __init__(self,n):
		super(DNN,self).__init__()
		self.hidden1 = Linear(n,512)
		self.act1 = ReLU()
		kaiming_uniform_(self.hidden1.weight,nonlinearity='relu')
		self.hidden2 = Linear(512,64)
		self.act2 = ReLU()
		kaiming_uniform_(self.hidden2.weight,nonlinearity='relu')
		self.output = Linear(64,32)
		self.act3 = ReLU()
		kaiming_uniform_(self.output.weight,nonlinearity='relu')

	def forward(self,X):
		X = self.hidden1(X)
		X = self.act1(X)
		X = self.hidden2(X)
		X = self.act2(X)
		X = self.output(X)
		X = self.act3(X)
		return X

a = [i for i in range(20000)]
model = DNN(20000)
print(model.forward(torch.FloatTensor(a)))