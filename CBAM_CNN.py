import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from ResNet_hc import resnet50
from CBAM_ResNet import resnet50
from CBAM_ResNeXt import resnext50
from ResNeXt import Resnext
from torch.autograd import Variable
import matplotlib.pyplot as plt
import confusion_matrix
# import CBAM-ResNeXt
from torchvision.models import vgg16
import torchvision.models as models
from torchvision.models import alexnet
from sklearn.metrics import cohen_kappa_score

# Setting hyperparameters (global parameters)

BATCH_SIZE = 16  #Number of images entered into the network at one time
EPOCHS = 100      #Number of iterations
modellr = 1e-4     #learning rate
DEVICE = torch. device( 'cuda' if torch.cuda.is_available() else'cpu')          #Determine whether to use cpu or GPU for training
#
# 数据预处理

transform = transforms.Compose([
    transforms.Resize((224, 224)),      #Resizing for uniformity
    # transforms.RandomVerticalFlip(),
    # transforms.RandomCrop(50),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),   #Redefining Size
    transforms.ToTensor(),          #Convert to Tensor file
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])# Normalization transforms to (-1, 1); regularization
])






# retrieve data
dataset_train = datasets.ImageFolder(r'D:\ResNet-master\Data\train', transform)  #Training dataset   Data enhancement
#print(dataset_train.imgs)
# The label of the corresponding folder
print(dataset_train.class_to_idx)     #The index of the category corresponding to the target returned without any conversion.
dataset_test = datasets.ImageFolder(r'D:\ResNet-master\Data\val', transform_test)  #Test Data Set
# The label of the corresponding folder
print(dataset_test.class_to_idx)

# Import data
np.random.seed(1234)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)  #A loader for the training dataset that automatically splits the data into batch, with the order randomly disrupted
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)    #No need for disruption

# Instantiate the model and move it to the GPU
criterion = nn.CrossEntropyLoss()   #Cross Entropy Loss Function


# #Method 1
# model = torchvision.models.resnext50_32x4d(pretrained=False)  #If using a pre-trained model, just change False to True:True/False
model = torchvision.models.resnet50(pretrained=False)
# ResNext,resnet,
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)  #Set category to 6
#
#
# model = torchvision.models.alexnet(pretrained=False)
# model = torchvision.models.resnet50(pretrained=False)
# model = torchvision.models.squeezenet1_0(pretrained=False)
# model = torchvision.models.densenet169(pretrained=False)
# model = torchvision.models.densenet121(pretrained=False)
# model = torchvision.models.inception_v3(pretrained=False)
# model = torchvision.models.googlenet(pretrained=False)
# model = torchvision.models.shufflenet_v2_x1_0(pretrained=False)
# model = torchvision.models.mobilenet_v2(pretrained=False)
# model = torchvision.models.resnext50_32x4d(pretrained=False)
# model = torchvision.models.mnasnet1_0(pretrained=False)
# # vgg16
# model_path = 'D:\ResNet-master\model/vgg16-397923af.pth'  #
# model = vgg16()
# num_ftrs = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(num_ftrs, 6)  #将类别设置为6
# # VIT
# num_ftrs = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(num_ftrs, 6)

# # alexnet
# model_path = 'D:\ResNet-master\model/alexnet-owt-7be5be79.pth'  #
# model = alexnet()
# num_ftrs = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(num_ftrs, 6)
# model.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, 6),
#         )
## mnasnet1_0
# model.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, 6),
#         )

##inception_v3
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 6)

# #densenet
# num_ftrs = model.classifier.in_features
# model.classifier = nn.Linear(num_ftrs, 6)


# # #Method 2 ResNet_hc (Add CBAM only)
# #
# # # model_path = 'D:\ResNet-master\model\Pre-training model parameters/resnet50-19c8e357.pth'  # Pre-training parameter positions own rewritten network
# model_path = 'D:\ResNet-master\model\Pre-training model parameters/resnext50_32x4d-7cdf4587.pth'  # Pre-training parameter positions own rewritten network
# # # model = resnet50()   #key
# model = resnext50()
# model_dict = model.state_dict()  # Network layer parameters
# #
# # Pre-training parameters to be loaded
# pretrained_dict = torch.load(model_path)#['state_dict']  # torch.load gets is a dictionary, we need the parameters under state_dict
# pretrained_dict = {k.replace('module.', ''): v for k, v in
#                    pretrained_dict.items()}  # Because pretrained_dict gets module.conv1.weight, but the mod you build yourself has no module, just conv1.weight, so rewrite it.
#
# # Remove what the mod does not have in pretrained_dict.items()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}#and k not in ['classifier.1.weight', 'classifier.1.bias']}  # Keep only the parameters in the pre-trained model that the self-built model has
# model_dict.update(pretrained_dict)  # Take the pre-trained values and update them into your own model's dict
# model.load_state_dict(model_dict)
#
# model.load_state_dict(torch.load(model_path), strict=False) # model loads the data in the dict and updates the initial values of the network
#
# in_channel = model.fc.in_features  #key
# resnext50.fc = nn.Linear(in_channel, 6)# Full connectivity layer
# # resnet50.fc = nn.Linear(in_channel, 6)# Fully-connected layer key

# #Method 3  CBAM_ResNet(CBAM and transfer learning)
# # model_path = 'D:\ResNet-master\model\Pre-training model parameters/resnet50-19c8e357.pth'  # pre-training parameters
# model_path = 'D:\ResNet-master\model\Pre-training model parameters/resnext50_32x4d-7cdf4587.pth'  # Pre-training parameter positions own rewritten network
# # model = resnet50()
# model = resnext50()
# model_dict = model.state_dict()  # Network layer parameters
# model.load_state_dict(torch.load(model_path), strict=False) # model loads the data in the dict and updates the initial values of the network
# in_channel = model.fc.in_features
# # resnet50.fc = nn.Linear(in_channel, 6)# Full connectivity layer
# resnext50.fc = nn.Linear(in_channel, 6)# Full connectivity layer


model.to(DEVICE)   #Placement of models into DEVICE
# Choose the simple and violent Adam optimizer with the learning rate tuned low
optimizer = optim.Adam(model.parameters(), lr=modellr)




def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs
(Setting the learning rate to an initial LR of 10 decays per 30 calendar elements)"""
    modellrnew = modellr * (0.1 ** (epoch // 50))
   # print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


# 定义训练过程

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        # enumerate is a built-in python function that gets both the index and the data, and the batch_idx parameter is used as the index of the output "precision/loss".
        data, target = Variable(data).to(device), Variable(target).to(device)   # Convert the data format with Variable and add it to the memory.
        output = model(data)    # Inputting data into the CNN network net
        # output, hidden = model(data) # This sentence applies to the inception_v3 network model
        loss = criterion(output, target)      # Calculation of the value of the loss
        optimizer.zero_grad()         # The gradient is set to zero because the gradient during backpropagation is added to the gradient of the loop one time
        loss.backward()               # Loss backpropagation
        optimizer.step()               # Parameter update after backpropagation
        print_loss = loss.data.item()
        # Getting the values of the elements of an elemental tensor is specifically used to convert a zero-dimensional tensor into a floating-point number, e.g., to calculate the value of loss, accuracy

        sum_loss += print_loss                # Loss accumulation
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))

    ave_loss = sum_loss / len(train_loader)
    print('epoch:{}'.format(epoch))
    print('train: avg_loss:{}'.format(ave_loss))   # Loss rate per iteration

# verification process
def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    #print(total_num, len(test_loader))
    pred_sum = np.zeros(0, dtype=int)
    target_sum = np.zeros(0, dtype=int)
    with torch.no_grad():
        for data, target in test_loader:

            data, target = Variable(data).to(device), Variable(target).to(device)
            target_sum = np.concatenate([target_sum, target.cpu().numpy()])
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)   # Calculate the number of correct predictions
            pred_sum = np.concatenate([pred_sum, pred.cpu().numpy()])
            print_loss = loss.data.item()
            test_loss += print_loss                      # Sum of total loss ratio

            # plt.figure()
            # confusion_matrix.plot_confusion_matrix(target_sum, pred_sum, normalize=True)
            # plt.show()
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        print('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))
        cm = confusion_matrix.plot_confusion_matrix(target_sum, pred_sum, normalize=True) # , normalize=True
        print(cm)
        print('Kappa:', cohen_kappa_score(target_sum, pred_sum))
        kp = cohen_kappa_score(target_sum, pred_sum)
        print(kp)
        # Updating the Optimal Confusion Matrix
        global ACC
        global CM
        global KP
        global Target
        global Pred
        if acc > ACC:
            ACC = acc
            CM = cm
            KP = kp
            Target = target_sum
            Pred = pred_sum


# Train
ACC = 0
CM = None
CM = 0
Target = None
Pred = None
for epoch in range(1, EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)
    train(model, DEVICE, train_loader, optimizer, epoch)
    val(model, DEVICE, test_loader)

# # Plotting Accuracy Curves
# plt.plot(EPOCHS, acc)
# plt.plot(EPOCHS, val_acc)
# plt.title('Training and validation accuracy')
# plt.legend(('Training accuracy', 'validation accuracy'))
# plt.figure()
#
# # Plotting loss curves
# plt.plot(EPOCHS, loss)
# plt.plot(EPOCHS, val_loss)
# plt.legend(('Training loss', 'validation loss'))
# plt.title('Training and validation loss')
# plt.show()

# Mapping the Optimal Confusion Matrix
print("the Optimal Confusion Matrix:")
print(ACC)
print(CM)
print(KP)
plt.close("all")
plt.figure()
confusion_matrix.plot_confusion_matrix(Target, Pred, normalize=True)#
plt.show()

# torch.save(model, 'D:/ResNet-master/CBAM-ResNeXt_model.pth')
