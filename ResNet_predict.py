import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.autograd import Variable
import os
from PIL import Image
import glob

classes = ('Camphor Tree','Glossy Privet',
           'Phoenix Tree','Pine Tree',
           'Photinia Serrulata', 'Sabina Chinensis')

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("CBAM-ResNeXt_model.pth")
model.eval()
model.to(DEVICE)

dataset_test = datasets.ImageFolder('Predict/png', transform_test)
print(len(dataset_test))


for index in range(len(dataset_test)):
    item = dataset_test[index]
    img, label = item
    img.unsqueeze_(0)
    data = Variable(img).to(DEVICE)
    output = model(data)
    _, pred = torch.max(output.data, 1)
    print('Image Name:{},predict:{}'.format(dataset_test.imgs[index][0], classes[pred.data.item()]))
    index += 1



