from torch.profiler import profile, record_function, ProfilerActivity
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

class LeNet(nn.Module):
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

class MLP(nn.Module):
    def __init__(self, n_hidden_layers, hidden_neurons, input_size, n_classes):
        super().__init__()

        self.fc_layers = nn.ModuleList()
        for i in range(n_hidden_layers+1):
            input_neurons = input_size if i == 0 else hidden_neurons[i-1]
            output_neurons = n_classes if i == n_hidden_layers else hidden_neurons[i]
            fc_layer = nn.Linear(input_neurons, output_neurons)
            self.fc_layers.append(fc_layer)

    def forward(self, x):
        x = torch.flatten(x, 1)
        for i, fc_layer in enumerate(self.fc_layers):
            x = fc_layer(x)

            if i < len(self.fc_layers) - 1:
                x = F.relu(x)

        return x

def train_lenet(model_params, model_name, device, epochs):
    model_path = MODEL_PATH + model_name + '.pth'
    net = LeNet(**model_params)

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

    return net

def train_mlp(model_params, model_name, device, epochs):
    model_path = MODEL_PATH + model_name + '.pth'
    net = MLP(**model_params)

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

    return MLP

if __name__ == "__main__":
    MODEL_PATH = './model_mnist/'
    RECORD_PATH = './records_mnist/'

    device = torch.device('cuda')

    transform = transforms.Compose([transforms.ToTensor()])
    batch_size = 4

    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

    # Load dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load testset
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    params_mlp1 = {"n_hidden_layers": 1, "hidden_neurons": [], "input_size": trainset[0][0].shape[-1]**2, "n_classes": 10}

    params_mlp2 = {"n_hidden_layers": 2, "hidden_neurons": [300, 100], "input_size": trainset[0][0].shape[-1]**2, "n_classes": 10}

    name_mlp2 = "MLP2_"+str(params_mlp2["hidden_neurons"][0])+str(params_mlp2["hidden_neurons"][1])

    params_lenet5 = {'n_conv': 2, 'n_fc': 3, 'conv_ch': [6, 16], 'filter_size': [5, 5], 'fc_size': [120, 84], 'pooling_size': 2, 'input_size': trainset[0][0].shape[-1], 'input_channels': 1, 'n_classes': len(classes), 'activation_fn': F.relu}

    name_lenet5 = "LeNet-5"

    for neurons in range(30, 301, 30):
        params_mlp1["hidden_neurons"] = [neurons]
        name_mlp1 = "MLP1" + str(neurons)
        train_mlp(params_mlp1, name, device, 20)

    train_mlp(params_mlp2, name_mlp2, device, 100)

    lenet5_model = train_lenet(params_lenet5, name_lenet5, device, 0)

    data = list(testloader)[0]
    inputs = data[0].to(device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True
    ) as prof:
        lenet5_model(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
