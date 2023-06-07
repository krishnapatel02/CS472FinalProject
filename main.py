from make_datasets import *
import torchvision.models as models
from alexnet import *
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from runModel import trainAndEval
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class args():
    def __init__(self, epochs, dropout, batch_size, lr, optim, loss_fn, hidden_layers):
        self.epochs = epochs
        self.dropout = dropout
        self.numClasses = 5
        self.batch_size = batch_size
        self.learning_rate = lr
        self.optim = optim
        self.hidden_layers = hidden_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.loss_fn = loss_fn





arguments = args(
    epochs=5,
    dropout= .2,
    batch_size=16,
    lr=0.001,
    optim=optim.Adadelta,
    loss_fn=torch.nn.CrossEntropyLoss,
    hidden_layers= 0
)


model = AlexNet_NoPerceptron(arguments)


train_data_loader = DataLoader(dataset, batch_size=arguments.batch_size, sampler=train_sampler)
test_data_loader = DataLoader(dataset, batch_size=arguments.batch_size, sampler=test_sampler)
validation_data_loader = DataLoader(dataset, batch_size=arguments.batch_size, sampler=validation_sampler)


print(model)
model.to(device)
trainAndEval(arguments, model, train_data_loader, test_data_loader, validation_data_loader)