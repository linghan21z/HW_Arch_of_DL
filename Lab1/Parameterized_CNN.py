import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import json

class ConvNet(nn.Module):
    def __init__(self, n_conv, n_fc, conv_ch, filter_size, fc_size, pooling_size, input_size, input_channels, n_classes, activation_fn):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        fc_input_size = input_size

        self.pooling_size = pooling_size
        if(pooling_size > 1):
            self.has_pooling = True
        else:
            self.has_pooling = False

        for i in range(n_conv):
            if(i == 0):
                input_channel = input_channels
            else:
                input_channel = conv_ch[i-1]
            conv_layer = nn.Conv2d(input_channel, conv_ch[i], filter_size[i])
            self.conv_layers.append(conv_layer)
            fc_input_size = fc_input_size - filter_size[i] + 1
            if(self.has_pooling):
                fc_input_size = (fc_input_size // pooling_size) if (fc_input_size % pooling_size == 0) else (fc_input_size // pooling_size + 1)

        self.fc_layers = nn.ModuleList()
        fc_input_size = conv_ch[-1] * fc_input_size * fc_input_size
        self.fc_layers.append(nn.Linear(fc_input_size, fc_size[0]))
        for i in range(1, n_fc-1):
            fc_layer = nn.Linear(fc_size[i-1], fc_size[i])
            self.fc_layers.append(fc_layer)

        self.output_layer = nn.Linear(fc_size[-1], n_classes)
        self.activation_fn = activation_fn

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.activation_fn(conv(x))
            # print(f"After conv the shape of x is: {x.shape}")
            if(self.has_pooling):
                pooling = nn.MaxPool2d(self.pooling_size, self.pooling_size, padding=x.shape[-1]%self.pooling_size)
                x = pooling(x)
            # print(f"After pooling the shape of x is: {x.shape}")

        x = torch.flatten(x, 1)

        # print(f"After flatten the shape of x is: {x.shape}")

        for fc in self.fc_layers:
            x = self.activation_fn(fc(x))

        return self.output_layer(x)

def train(model_params, model_name, device, epochs):
    model_path = MODEL_PATH + model_name + '.pth'
    net = ConvNet(**model_params)

    # load model state
    try:
        net.load_state_dict(torch.load(model_path))
        print("Model state loaded successfully.")
    except FileNotFoundError:
        os.makedirs(MODEL_PATH, exist_ok=True)
        print(f"No saved model state found at '{model_path}'.")

    net.to(device)

    # load record
    record_path = RECORD_PATH + model_name + '.json'
    try:
        with open(record_path, 'r') as file:
            record = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        os.makedirs(RECORD_PATH, exist_ok=True)
        record = {"name": model_name, "epochs": 0, "training_records": []}

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                # torch.save(net.state_dict(), PATH)

    print('Finished Training')

    # save current epochs
    record["epochs"] = record["epochs"] + epochs

    # save model
    torch.save(net.state_dict(), model_path)

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    test_result = {}

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        test_result[classname] = accuracy

    record["training_records"].append({"training_epoch": record["epochs"], "accuracy": test_result})

    with open(record_path, 'w+') as file:
        json.dump(record, file, indent=4)

if __name__ == "__main__":
    MODEL_PATH = './model_CIFAR10_10iters/'
    RECORD_PATH = './records_CIFAR10_10iters/'

    device = torch.device('cuda')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4
    # Load trainset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load testset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    conv_params_original = {'n_conv': 2, 'n_fc': 3, 'conv_ch': [6, 16], 'filter_size': [5, 5], 'fc_size': [120, 84], 'pooling_size': 2, 'input_size': trainset[0][0].shape[2], 'input_channels': 3, 'n_classes': len(classes), 'activation_fn': F.relu}

    conv_params_c1_8 = {'n_conv': 2, 'n_fc': 3, 'conv_ch': [8, 16], 'filter_size': [5, 5], 'fc_size': [120, 84], 'pooling_size': 2, 'input_size': trainset[0][0].shape[2], 'input_channels': 3, 'n_classes': len(classes), 'activation_fn': F.relu}

    conv_params_c1_10 = {'n_conv': 2, 'n_fc': 3, 'conv_ch': [10, 16], 'filter_size': [5, 5], 'fc_size': [120, 84], 'pooling_size': 2, 'input_size': trainset[0][0].shape[2], 'input_channels': 3, 'n_classes': len(classes), 'activation_fn': F.relu}

    conv_params_c1_12 = {'n_conv': 2, 'n_fc': 3, 'conv_ch': [12, 16], 'filter_size': [5, 5], 'fc_size': [120, 84], 'pooling_size': 2, 'input_size': trainset[0][0].shape[2], 'input_channels': 3, 'n_classes': len(classes), 'activation_fn': F.relu}

    conv_params_c3_20 = {'n_conv': 2, 'n_fc': 3, 'conv_ch': [6, 20], 'filter_size': [5, 5], 'fc_size': [120, 84], 'pooling_size': 2, 'input_size': trainset[0][0].shape[2], 'input_channels': 3, 'n_classes': len(classes), 'activation_fn': F.relu}

    conv_params_c3_24 = {'n_conv': 2, 'n_fc': 3, 'conv_ch': [6, 24], 'filter_size': [5, 5], 'fc_size': [120, 84], 'pooling_size': 2, 'input_size': trainset[0][0].shape[2], 'input_channels': 3, 'n_classes': len(classes), 'activation_fn': F.relu}

    conv_params_c3_28 = {'n_conv': 2, 'n_fc': 3, 'conv_ch': [6, 28], 'filter_size': [5, 5], 'fc_size': [120, 84], 'pooling_size': 2, 'input_size': trainset[0][0].shape[2], 'input_channels': 3, 'n_classes': len(classes), 'activation_fn': F.relu}

    conv_params_9_layers = {'n_conv': 3, 'n_fc': 3, 'conv_ch': [6, 16, 26], 'filter_size': [5, 5, 2], 'fc_size': [120, 84], 'pooling_size': 2, 'input_size': trainset[0][0].shape[2], 'input_channels': 3, 'n_classes': len(classes), 'activation_fn': F.relu}

    conv_params_11_layers = {'n_conv': 4, 'n_fc': 3, 'conv_ch': [6, 16, 26, 36], 'filter_size': [2, 2, 2, 2], 'fc_size': [120, 84], 'pooling_size': 2, 'input_size': trainset[0][0].shape[2], 'input_channels': 3, 'n_classes': len(classes), 'activation_fn': F.relu}

    conv_params_13_layers = {'n_conv': 5, 'n_fc': 3, 'conv_ch': [6, 16, 26, 36, 50], 'filter_size': [2, 2, 2, 2, 2], 'fc_size': [120, 84], 'pooling_size': 2, 'input_size': trainset[0][0].shape[2], 'input_channels': 3, 'n_classes': len(classes), 'activation_fn': F.relu}

    conv_params_sigmoid = {'n_conv': 2, 'n_fc': 3, 'conv_ch': [6, 16], 'filter_size': [5, 5], 'fc_size': [120, 84], 'pooling_size': 2, 'input_size': trainset[0][0].shape[2], 'input_channels': 3, 'n_classes': len(classes), 'activation_fn': F.sigmoid}

    conv_params_tanh = {'n_conv': 2, 'n_fc': 3, 'conv_ch': [6, 16], 'filter_size': [5, 5], 'fc_size': [120, 84], 'pooling_size': 2, 'input_size': trainset[0][0].shape[2], 'input_channels': 3, 'n_classes': len(classes), 'activation_fn': F.tanh}

    models_params = [conv_params_original, conv_params_c1_8, conv_params_c1_10, conv_params_c1_12, conv_params_c3_20, conv_params_c3_24, conv_params_c3_28, conv_params_9_layers, conv_params_11_layers, conv_params_13_layers, conv_params_sigmoid, conv_params_tanh]

    models_names = ["conv_params_original", "conv_params_c1_8", "conv_params_c1_10", "conv_params_c1_12", "conv_params_c3_20", "conv_params_c3_24", "conv_params_c3_28", "conv_params_9_layers", "conv_params_11_layers", "conv_params_13_layers", "conv_params_sigmoid", "conv_params_tanh"]

    for i in range(10):
      for param, name in zip(models_params, models_names):
          print("Model <"+name+"> starts training")
          train(param, name, device, 1)
          print("Model <"+name+"> finishes training")
