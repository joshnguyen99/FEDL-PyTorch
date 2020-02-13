import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FedModel import Net, MCLR
from FedOptimizer import MySGD, FEDLOptimizer

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1


class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, numeric_id, dataset, model, batch_size, learning_rate,
                 local_epochs, optimizer):
        # Set up the main attributes
        self.id = "f_%05d" % numeric_id
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.loss = nn.NLLLoss()
        self.model = MCLR()

        # Get the dataset and divide it in batches
        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = \
            self.get_data(self.id, dataset)
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]
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

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0]).item()
            print(self.id + ", Accuracy:", test_acc)
        return test_acc / self.batch_size

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


class UserAVG(User):
    """
    User in FedAvg.
    """
    def __init__(self, numeric_id, dataset, model, batch_size, learning_rate,
                 local_epochs, optimizer):
        super().__init__(numeric_id, dataset, model, batch_size, learning_rate,
                         local_epochs, optimizer)

        if optimizer == "SGD":
            self.optimizer = MySGD(self.model.parameters(), lr=0.01)

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param = new_param.clone().requires_grad_(True)
            # old_param.data = new_param.data.clone()
        self.optimizer = MySGD(self.model.parameters(), lr=0.01)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = F.nll_loss(output, y)
                loss.backward()
                self.optimizer.step()
                LOSS += loss.item()
        return LOSS / self.train_samples


class UserFEDL(User):
    def __init__(self, numeric_id, dataset, model, batch_size, learning_rate,
                 local_epochs, optimizer, eta):
        super().__init__(numeric_id, dataset, model, batch_size, learning_rate,
                         local_epochs, optimizer)

        self.eta = eta

        self.pre_grads, self.server_grads = None, None

        self.optimizer = FEDLOptimizer(self.model.parameters(),
                                       lr=self.learning_rate,
                                       server_grads=self.server_grads,
                                       pre_grads=self.pre_grads,
                                       eta=self.eta)
        self.optimizer.zero_grad()
        loss = F.nll_loss(self.model(self.X_train), self.y_train)
        loss.backward()

        self.server_model = MCLR()
        # self.server_loss = F.nll_loss(self.server_model(self.X_train), self.y_train)
        # self.server_loss.backward()

        self.save_previous_grads()

    def set_parameters(self, model):

        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param = new_param.clone().requires_grad_(True)
            # old_param.data = new_param.data.clone()
        self.optimizer = FEDLOptimizer(self.model.parameters(),
                                       lr=self.learning_rate,
                                       server_grads=self.server_grads,
                                       pre_grads=self.pre_grads,
                                       eta=self.eta)

        # for old_param, new_param in zip(self.model.parameters(), model.parameters()):
        #     old_param.data = new_param.data.clone()

        # Find F_n(w^{t-1})
        self.optimizer.zero_grad()
        output = self.model(self.X_train)
        server_loss = F.nll_loss(output, self.y_train)
        server_loss.backward()
        self.save_previous_grads()

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        for epoch in range(self.local_epochs):
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = F.nll_loss(output, y)
                LOSS += loss.item()
                loss.backward()
                self.optimizer.step()
        return LOSS / self.train_samples

    def save_previous_grads(self):
        pre_grads = []
        for param in self.server_model.parameters():
            if param.grad is not None:
                pre_grads.append(param.grad.data.clone())
            else:
                pre_grads.append(torch.zeros_like(param.data))
        self.pre_grads = pre_grads
        self.optimizer.pre_grads = pre_grads

    def save_server_grads(self, server_grads):
        self.server_grads = server_grads
        self.optimizer.server_grads = server_grads
