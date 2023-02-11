import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

image = Image.open(str('./dog.jpg'))
model = models.resnet50(pretrained=True)

#append all the conv layers and their respective wights to the list
model_weights =[]
conv_layers = []
model_children = list(model.children())
counter = 0
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

output=model(image)
print(output.argmax())

def visual_feature_map(image):
    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    processed = []
    for feature_map in outputs:
        #print(feature_map.shape)
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    fig = plt.figure(figsize=(50, 50))
    for i in range(len(processed)): 
        a = fig.add_subplot(7, 7, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0]+' No.'+str(i)+' size '+str(processed[i].shape), fontsize=30)
    plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
visual_feature_map(image)  

def plot_weights(weights,idx):
    t = weights[idx].cpu().detach()
    fig = plt.figure(figsize=(50,50))
    #looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(8,8,i+1)
        npimg = np.array(t[i].numpy(), np.float32)
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title('filter No.'+str(i+1),fontsize=30)
    plt.savefig('filters of layer '+str(idx)+' .png' , bbox_inches='tight')  
#plot_weights(model_weights,0)
