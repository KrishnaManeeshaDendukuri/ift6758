import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_dims = [16,10,6]
lr = 0.001
epochs = 4000
n_feats = 32


# model class
class binary_classifier(nn.Module):

    def __init__(self):
        super(binary_classifier,self).__init__()
        self.layer1 = nn.Linear(n_feats, hidden_dims[0])
        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.layer3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.layer4 = nn.Linear(hidden_dims[2], 2)

    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x

    #This function takes an input and predicts the class, (0 or 1)        
    def predict_class(self,x):
        #Apply softmax to output. 
        pred = F.softmax(self.forward(x), dim=1)
        ans = []
        #Pick the class with maximum weight
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

    def predict_probs(self,x):
        #Apply softmax to output. 
        pred = F.softmax(self.forward(x), dim=1)
        return pred