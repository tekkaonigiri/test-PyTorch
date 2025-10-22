"""
MNIST - Modified National Institute of Standards and Technology
https://www.kaggle.com/code/geekysaint/solving-mnist-using-pytorch
    using the handwritten digits database as our dataset
"""

import torch
import torchvision ## contains some utilities for working w the image data
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F



# loading mnist dataset
dataset = MNIST(root = 'data/', download = True)
print(len(dataset))

image, label = dataset[10]
plt.imshow(image, cmap = 'gray')
print('Label:', label)
"""
need to convert images into tensors so pytorch can understand
    do this by specifying a transform while creating dataset
    pytorch allow us to specify one or more transformation func
        which are applied to imgs as they're loaded
"""

# loading mnist data w transformation applied while loading
mnist_dataset = MNIST(root = 'data/', train = True, transform = transforms.ToTensor())
print(mnist_dataset)

image_tensor, label = mnist_dataset[0]
print(image_tensor.shape, label)
    # is now converted to a 28 by 28 tensor
    # images in mnist data set are grayscale -- others have RGB

print(image_tensor[:, 10:15, 10:15])
print(torch.max(image_tensor), torch.min(image_tensor))
    # values range from 0 (black) to 1 (white)

plt.imshow(image_tensor[0, 10:15, 10:15], cmap = 'gray')
    # plot tensor as an iamge



""" datasets -- training + validation """

train_data, validation_data = random_split(mnist_dataset, [50000, 10000])
print("length of Train Datasets: ", len(train_data))
print("length of Validation Datasets: ", len(validation_data))
    # print length of train + validation datasets

# DataLoaders to load data in batches
batch_size = 128    # why are we using size 128
train_loader = DataLoader(train_data, batch_size, shuffle = True)
    # shuffle = True so batches generated in each epoch are diff
    # randomization helps in generalizing + speeding up process
val_loader = DataLoader(validation_data, batch_size, shuffle = False)
    # no need to shuffle images since validation only used for evaluating model



""" model, nn.Linear """

import torch.nn as nn
input_size = 28 * 28
num_classes = 10

"""         gemini says to delete this part since its the wrong model
        - shape mismatch that prevented mat1 and mat2 shapes from being multiplied
    due to premature testing
        - deleting this ensured that the next time variable model was assigned, its
    created from the correct class definition (line 139, model = MnistModel())

# logistic regression model
model = nn.Linear(input_size, num_classes)
print(model.weight.shape)
print(model.weight)
print(model.bias.shape)
print(model.bias)

for images, labels in train_loader:
    print(labels)
    print(images.shape)
    outputs = model(images)     # crashing here bc using wrong model??
    break
"""

# define logistic model
class MnistModel(nn.Module):
    # instantiate weights + biases using nn.Linear
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
            # initialize linear
    
    # invoked when passing a batch of inputs to model
    # flatten out input tensor, then pass to self.linear
    def forward(self, xb):
        xb = xb.flatten(start_dim = 1)
            # FLATTEN image tensor from (batch, 1, 28, 28) to (batch, 784)
            # length along 2D is 28*28 = 784
        out = self.linear(xb)
            # matrix multiplication
        return(out)
    

    # adding cross-entropy here
    # refer to section for cross-entropy where i add more functions
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # generation predictions
        loss = F.cross_entropy(out, labels) # calculate the loss
        return(loss)
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return({'val_loss':loss, 'val_acc':acc})
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()

        return({'val_loss':epoch_loss.item(), 'val_acc':epoch_acc.item()})
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    # refer to section for cross-entropy where i add more functions

model = MnistModel()
print(model.linear.weight.shape, model.linear.bias.shape)
    # .lnear attribute now includes weight and bias attributes
list(model.parameters())
    # returns a list containing weights + biases
    # can be used by a pytorch optimizer

"""
for each 100 input images, get 10 outputs (one for each class)
outputs represent probabilities
output row lie btwn 0 to 1, add up to 1
"""
for images, labels in train_loader:
    outputs = model(images)
    break
print('outputs shape: ', outputs.shape)
print('Sample outputs: \n', outputs[:2].data)



""" softmax """

# apply softmax for each output row
probs = F.softmax(outputs, dim = 1)
print("Sample probabilities:\n", probs[:2].data)
    # checking at sample probs

print("\n")
print("Sum: ", torch.sum(probs[0]).item())
    # add up probs of an output row
max_probs, preds = torch.max(probs, dim = 1)
print("\n")
print(preds)
print("\n")
print(max_probs)

labels



""" evaluation metric + loss function """

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
        # compare preds and labels -- element-wise comparison
        # divide by total number of predictions to get average
    """
    good evaluation metric but bad for loss function
    - does not take into account the actual probs predicted by model --> cannot
        provide sufficient feedback for incremental improvements
    """
    # "==" in torch.sum performs element-wise comparison of two tensors w same shape

print("Accuracy: ", accuracy(outputs, labels))
print("\n")
loss_fn = F.cross_entropy
print("Loss Function: ", loss_fn)
print("\n")
# loss for the current batch
loss = loss_fn(outputs, labels)
print(loss)

# cross-entropy -- added to the class MnistModel
def evaluate(mode, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return(model.validation_epoch_end(outputs))

def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):

        # training phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    
    return(history)

result0 = evaluate(model, val_loader)
result0
# initial accuracy is around 8%



"""training for 5 epochs"""
history1 = fit(5, 0.001, model, train_loader, val_loader)
history2 = fit(5, 0.001, model, train_loader, val_loader)
history3 = fit(5, 0.001, model, train_loader, val_loader)
history4 = fit(5, 0.001, model, train_loader, val_loader)

# replace values w ur result
history = [result0] + history1 + history2 + history3 + history4
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')

# testing w individual images
# define the test dataset
test_dataset = MNIST(root = 'data/', train = False, transform = transforms.ToTensor())

img, label = test_dataset[0]
plt.imshow(img[0], cmap = 'gray')
print("shape: ", img.shape)
print('Label: ', label)

print(img.unsqueeze(0).shape)
print(img.shape)

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim = 1)
    return(preds[0].item())

img, label = test_dataset[0]
plt.imshow(img[0], cmap = 'gray')
print('Label:', label, ', Predicted :', predict_image(img,model))

img, label = test_dataset[9]
plt.imshow(img[0], cmap = 'gray')
print('Label:', label, ', Predicted :', predict_image(img,model))

img, label = test_dataset[25]
plt.imshow(img[0], cmap = 'gray')
print('Label:', label, ', Predicted :', predict_image(img,model))

img, label = test_dataset[5000]
plt.imshow(img[0], cmap = 'gray')
print('Label:', label, ', Predicted :', predict_image(img,model))

test_loader = DataLoader(test_dataset, batch_size = 256)
result = evaluate(model, test_loader)
result


# saving + loading model
torch.save(model.state_dict(), 'mnist-logistic.pth')
    # .stage_dict returns an OrderedDict w all the weights +
        # bias matrices mapped to the correct attributes of model
model.state.dict()

model2 = MnistModel()
model2.load_state_dict(torch.load('mnist-logistic.pth'))
model2.state_dict()

test_loader = DataLoader(test_dataset, batch_size = 256)
result = evaluate(model2, test_loader)
result