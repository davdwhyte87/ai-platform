import torch
from torchvision import datasets, transforms
import argparse
from torch import nn, optim
import torch.nn.functional as F
import mlflow
from tensorboardX import SummaryWriter
import os
import tempfile

parser = argparse.ArgumentParser(description='Fashion mnist NN')
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000,
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--enable-cuda', type=str, choices=['True', 'False'], default='True',
                    help='enables or disables CUDA training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100,
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

enable_cuda_flag = True if args.enable_cuda == 'True' else False

args.cuda = enable_cuda_flag and torch.cuda.is_available()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

# test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True) 

class Classifier(nn.Module):
  def __init__(self):
    super(Classifier,self).__init__()
    self.layer_1 = nn.Linear(784, 256)
    self.layer_2 = nn.Linear(256, 128)
    self.layer_3 = nn.Linear(128, 64)
    self.dropout = nn.Dropout(p=0.2)
    self.layer_4 = nn.Linear(64, 10)
        
  def forward(self, x):
    x = x.view(x.shape[0], -1)
    # print(x.shape)
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = F.relu(self.layer_3(x))
    x = F.log_softmax(self.layer_4(x), dim=1)
    return x

model = Classifier()
if args.cuda:
    model.cuda()
# gets an image batch 
# images, labels = next(iter(testloader))

# ps = model(images)

criterion = nn.NLLLoss()
optimizer = optim.Adam(params=model.parameters() ,lr=args.lr)
epochs = args.epochs
training_losses, test_loses = [], []
model.train()

def train(epochs):
  running_loss = 0
  for images, labels in trainloader:
    if args.cuda:
        images, labels = images.cuda(), labels.cuda()
    optimizer.zero_grad()
    log_ps = model(images)
    loss = criterion(log_ps, labels)
    loss.backward()
    optimizer.step()
    running_loss +=loss.item()
  training_losses.append(running_loss/len(trainloader))


def test(e):
  test_loss = 0
  accuracy = 0
  with torch.no_grad():
    model.eval()
    for images, labels in testloader:
      log_ps = model(images)
      test_loss += criterion(log_ps, labels)
      ps = torch.exp(log_ps)
      top_p, top_class = ps.topk(1, dim=1)
      equals = top_class == labels.view(*top_class.shape)
      accuracy += torch.mean(equals.type(torch.FloatTensor))
  model.train()
  test_loses.append(test_loss/len(testloader))
  # log data
  # mlflow.log_metric("Training loss", float(training_losses[-1]))
  # mlflow.log_metric("Test loss", float(test_loses[-1]))
  # mlflow.log_metric("Test Accuracy", accuracy/len(testloader))

  print("Epoch: {}/{}.. ".format(e+1, epochs),
          "Training Loss: {:.3f}.. ".format(training_losses[-1]),
          "Test Loss: {:.3f}.. ".format(test_loses[-1]),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

writer = None

def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)

with mlflow.start_run():
  for key, value in vars(args).items():
      mlflow.log_param(key, value)
  
  # Create a SummaryWriter to write TensorBoard events locally
  output_dir = dirpath = tempfile.mkdtemp()
  writer = SummaryWriter(output_dir)
  print("Writing TensorBoard events locally to %s\n" % output_dir)

  # Perform the training
  for e in range(epochs):
    train(e)
  print("Training loss", training_losses[-1])
  # testing
  test(e)

  # Upload the TensorBoard event logs as a run artifact
  print("Uploading TensorBoard events as a run artifact...")
  mlflow.log_artifacts(output_dir, artifact_path="events")
  print("\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s" %
      os.path.join(mlflow.get_artifact_uri(), "events"))
# plt.plot(training_losses, label='Training loss')
# plt.plot(test_loses, label='Validation loss')
# plt.legend(frameon=False)
# plt.show()