from apex import amp
import torch.cuda.amp
import torch
from torch import nn
import time

device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.u1_1 = nn.Parameter(torch.randn(16,16,16))
        self.b2_1 = nn.Parameter(torch.randn(16,16,16))
        self.u2_1 = nn.Parameter(torch.randn(16,16,16))
        self.b1_1 = nn.Parameter(torch.randn(16,16))
        self.u3_1 = nn.Parameter(torch.randn(16,16,16))
        self.b1 = nn.Parameter(torch.randn(16,16,16))

    def forward(self, x):
        batch = x.shape[0]
        x = torch.reshape(x,(batch,16,16,16))
        x = torch.einsum('zabc,ade->zbcde',(x,self.u1_1))
        x = torch.einsum('zabcd,def->zabcef',(x,self.b2_1))
        x = torch.einsum('zabcde,afd->zbcef',(x,self.u2_1))
        x = torch.einsum('zabcd,ce->zabde',(x,self.b1_1))
        x = torch.einsum('zabcd,aed->zbce',(x,self.u3_1))
        x = x + self.b1
        x = torch.reshape(x,(batch,16*16*16))
        #x = self.fc1(x)

        return x



input = torch.randn(182,16,16,16).to(device)
model = FCNet()
model = model.to(device)

torch.cuda.synchronize()
start = time.time()

for i in range(20):
    with torch.cuda.amp.autocast():
        out = model(input)
# for i in range(10):
#     out = model(input)
#     print(i)
#print(out[1])
torch.cuda.synchronize()
end = time.time()

print("cost time is:",end - start)