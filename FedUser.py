import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from FedModel import Net
from FedOptimizer import MySGD, FEDLOptimizer

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1


class UserAVG:
    def __init__(self, numeric_id, dataset, model, batch_size, learning_rate,
                 local_epochs, optimizer):
        self.id = "f_%05d" % numeric_id
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataset = dataset
        self.local_epochs = local_epochs

        X_train, y_train, X_test, y_test, self.train_samples, self.test_samples = \
            self.get_data(self.id, dataset)
        self.train_data = [(x, y) for x, y in zip(X_train, y_train)]
        self.test_data = [(x, y) for x, y in zip(X_test, y_test)]

        if model == "cnn":
            if self.model_exists():
                self.load_model()
            else:
                self.model = Net()
        self.loss = nn.NLLLoss()

        if optimizer == "SGD":
            self.optimizer = MySGD(self.model.parameters(), lr=0.01)
        self.trainloader = DataLoader(self.train_data, self.batch_size)
        self.testloader = DataLoader(self.test_data, self.test_samples)

    def get_data(self, id="", dataset="mnist"):
        train_path = os.path.join("datafed", self.dataset, "userstrain", id + ".json")
        test_path = os.path.join("datafed", self.dataset, "userstest", id + ".json")
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("User not detected.")

        with open(os.path.join(train_path), "r") as f_train:
            train = json.load(f_train)

        with open(os.path.join(test_path), "r") as f_test:
            test = json.load(f_test)

        X_train, y_train, X_test, y_test = train['X'], train['y'], test['X'], test['y']
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
        train_samples, test_samples = train["num_samples"], test["num_samples"]

        return X_train, y_train, X_test, y_test, train_samples, test_samples

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def set_parameters(self, new_params):
        for old_param, new_param in zip(self.model.parameters(), new_params):
            old_param = new_param.clone().requires_grad_(True)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = MySGD(self.model.parameters(), lr=0.01)

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = []
        self.model.train()
        for epoch in tqdm(range(1, self.local_epochs + 1), desc="Local Epoch"):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                # print(torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0])
                loss = F.nll_loss(output, y)
                loss.backward()
                self.optimizer.step()
                LOSS.append(loss.item())
        return LOSS

    def test(self):
        self.model.eval()
        for x, y in self.testloader:
            output = self.model(x)
            print(self.id + ", Accuracy:", torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0])

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        # assert (os.path.exists(os.path.join("models", "user_" + self.id + ".pt")))
        # self.model = torch.load(os.path.join("models", "user_" + self.id + ".pt"))
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))


    @staticmethod
    def model_exists():
        # return os.path.exists(os.path.join("models", "user_" + self.id + ".pt"))
        return os.path.exists(os.path.join("models", "server" + ".pt"))


class UserFEDL(UserAVG):
    def __init__(self, numeric_id, dataset, model, batch_size, learning_rate, local_epochs, optimizer):
        super().__init__(numeric_id, dataset, model, batch_size, learning_rate, local_epochs, optimizer)
        if model == "cnn":
            if self.model_exists():
                self.load_model()
            else:
                self.model = Net()
        self.loss = nn.NLLLoss()

        self.pre_grads, self.server_grads = None, None

        self.optimizer = FEDLOptimizer(self.model.parameters(),
                                       lr=self.learning_rate,
                                       server_grads=self.server_grads,
                                       pre_grads=self.pre_grads)
        self.save_previous_grads()

    def set_parameters(self, new_params):
        for old_param, new_param in zip(self.model.parameters(), new_params):
            old_param = new_param.clone().requires_grad_(True)
        self.optimizer = FEDLOptimizer(self.model.parameters(),
                                       lr=self.learning_rate,
                                       server_grads=self.server_grads,
                                       pre_grads=self.pre_grads)

    def train(self, epochs):
        LOSS = []
        self.save_previous_grads()
        self.model.train()
        for epoch in tqdm(range(1, self.local_epochs + 1), desc="Local Epoch"):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = F.nll_loss(output, y)
                loss.backward()
                self.optimizer.step()
                LOSS.append(loss.item())
        return LOSS

    def save_previous_grads(self):
        pre_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                pre_grads.append(param.grad)
            else:
                pre_grads.append(torch.zeros_like(param.data))
        self.pre_grads = pre_grads
        self.optimizer.pre_grads = pre_grads

    def save_server_grads(self, server_grads):
        self.server_grads = server_grads
        self.server_grads = server_grads
