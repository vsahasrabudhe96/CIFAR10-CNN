import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import sys
import cv2
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# converting it to a tensor and normalizing it with mean an standard deviation
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.CIFAR10(
    root='./model/data.cifar10',  # location of the dataset
    train=True,  # this is training data
    transform=transform,  # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
    download=True  # if you haven't had the dataset, this will automatically download it for you
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=128,
                               shuffle=True)  # setting the batch size to be 6000, and setting the shuffle parameter as true ensures that the iimages in the batches will be randomly shuffled

test_data = torchvision.datasets.CIFAR10(root='./model/data.cifar10/', train=False, transform=transform)

test_loader = Data.DataLoader(dataset=test_data, batch_size=128, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
           'truck')  # the 10 cimage classes are stored in the tuple
epochs = 10 # number of epochs for each the model will run
# num_examples = 20000
# num_test = 10000

class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10,self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool1 = nn.MaxPool2d(5, 1)
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128, 5)
        self.pool2 = nn.MaxPool2d(3,2)
        self.batch2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256,5)
        self.pool3 = nn.MaxPool2d(3,2)
        self.batch3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 2 * 2, 1024)
        #self.drop1 = nn.Dropout(p =  0.35)
        #self.batch4 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 600)
        #self.batch5 = nn.BatchNorm1d(600)
        self.fc3 = nn.Linear(600, 10)

    def forward(self, x):
        l1 = self.conv1(x)
        x = self.pool1(F.relu(self.batch1(l1)))
        x = self.pool2(F.relu(self.batch2(self.conv2(x))))
        x = self.pool3(F.relu(self.batch3(self.conv3(x))))
        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x,l1


model = CIFAR10()

optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.0005)  # using Adam optimizer and setting the learning rate as 0.0005(we can also use SGD and rmsprop)
loss_func = nn.CrossEntropyLoss()  # We make us cross entropy loss as the loss function(we can also make use of NLLLoss and log softmax)


#########TRAINING FUNCTION###########
def train():
    total_loss = 0
    total_train = 0
    correct_train = 0
    for images, labels in train_loader:
        model.train()  # setting the model in training mode
        optimizer.zero_grad()  # every time train we want to gradients to be set to zero
        output,layer1= model(images)  # making the forward pass through the model
        loss = loss_func(output, labels)
        # loss2 = loss_func(layer1,labels)
        # loss = loss1+loss2

        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()

        # accuracy
        _, predicted = torch.max(output.data, 1)  # we check the label which has maximum probability
        total_train += labels.size(0)
        correct_train += predicted.eq(labels.data).sum().item()
        train_accuracy = 100 * correct_train / total_train
    return (train_accuracy, loss.item())


def test(epochs):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs,layer1 = model(images)  # forward pass
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)  # to get the total number of labels
                correct += (
                            predicted == labels).sum().item()  # we check the amount of predicted and actual labels which are same and then sum them for calculating the accuracy
                acc = 100 * correct / total
    return acc




def predict(img_path):
    # Loading the model
    model = torch.load('./model/model.pt')  # loading the model from the model path
    # params = list(model.parameters())
    # l1 = params[0].size()
    # print(l1)
    # Loadind the test image
    from PIL import Image
    img = Image.open(img_path)
    #img = cv2.imread(img_path)

    img_re = img.resize((32, 32))

    with torch.no_grad():
        trans1 = transforms.ToTensor()
        img_tensor = trans1(img_re)  # shape = [3, 32, 32]

        # Image Transformation
        trans = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img_tensor = trans(img_tensor)

        single_image_batch = img_tensor.unsqueeze(0)  # shape = [1, 3, 32, 32]
        outputs,layer1 = model(single_image_batch)
        _, predicted = torch.max(outputs.data, 1)
        class_id = predicted[0].item()
        predicted_class = classes[predicted[0].item()]
        print("Predicted Class : {}".format(predicted_class))
        # #print('Output layer 1:')
        for i in range(layer1.size(1)):
            plt.subplot(6,6,i+1)
            #plt.title('Filter' + str(i))
            plt.axis('off')
            c = layer1[0,i,:,:].detach().numpy()
            plt.imshow(c,interpolation='nearest',cmap='gray')
        plt.savefig('CONV_rslt')

    plt.rcParams["figure.figsize"] = (2, 2)
    #plt.title(predicted_class)
    #plt.imshow(img)
    plt.show()




def main():
    print("Training for {} epochs".format(epochs))
    for epoch in range(epochs):
        (train_model, loss_tr) = train()
        test_model = test(epochs)
        print('Epoch {}, train Loss: {:.3f}'.format(epoch, loss_tr), "Training Accuracy {}:".format(train_model),
              'Test Accuracy {}'.format(test_model))
    print("Model successfully saved at ./model/")
    torch.save(model, './model/model.pt')  # saving the model to the mentioned model path


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Please enter more commands for training and testing e.g 'train','test'")
        sys.exit()

    elif len(sys.argv) == 2:
        try:
            assert sys.argv[1] == "train"
            main()
            sys.exit()
        except AssertionError:
            print('Please enter a valid command')
            sys.exit()

    elif len(sys.argv) == 3:
        try:
            assert sys.argv[1] == "test"
            if not sys.argv[2].split('.')[-1].endswith('png'):
                print('Only PNG files supported')
                sys.exit()
            else:
                predict(sys.argv[2])
        except AssertionError:
            print('Please enter a valid command')
            sys.exit()
