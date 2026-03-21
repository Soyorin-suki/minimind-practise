import torch
import torch.nn as nn

# dropout_layer = nn.Dropout(p=0.5)

# t1=torch.Tensor([1,2,3])
# t2=dropout_layer(t1)
# print(t2)

# layer = nn.Linear(in_features=3, out_features=5, bias=True)
# t1 = torch.Tensor([1,2,3])		# shape = (3)
# t2 = torch.Tensor([[1,2,3]])	# shape = (1,3)
# # 这里应用的w与b是随机的，真实训练中里会在optimizer上更新
# output2 = layer(t2)
# print(output2)
# # 线性变换，对应用的张量乘以一个w矩阵，然后+b

# t = torch.Tensor([[1,2,3,4,5,6],[7,8,9,10,11,12]])
# t1 = t.view(3,4)
# t2 = t.view(4,3)
# t3 = t.view(2,2,3)
# print(t1)
# print(t2)
# print(t3)
# 这里的t1,t2,t3和t是共享一块内存空间的，也就是他们只是引用

# t = torch.Tensor([[1,2,3],[4,5,6]])
# t = t.transpose(0,1)
# print(t)

# x = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
# print(torch.triu(x))
# # 掩码，将(0,0)对角线以下的全部置为0
# print(torch.triu(x,diagonal=1))
# # 将(0,1)对角线以下的位置全部置为0

