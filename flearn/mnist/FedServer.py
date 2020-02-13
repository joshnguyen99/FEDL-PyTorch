import torch
import os

from .FedUser import UserAVG, UserFEDL
from .FedModel import MCLR


class Server:
    def __init__(self, dataset, model, batch_size, learning_rate,
                 num_glob_iters, local_epochs, optimizer, num_users=100):

        # Set up the main attributes
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.users = []
        self.total_train_samples = 0
        self.model = MCLR()

        # Initialize the server's grads to zeros
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
            param.grad = torch.zeros_like(param.data)
        # self.send_parameters()

    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        for user in self.users:
            self.add_parameters(user, user.train_samples / self.total_train_samples)

    def test(self):
        for user in self.users:
            user.test()

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))


class ServerAVG(Server):
    def __init__(self, dataset, model, batch_size, learning_rate, num_glob_iters,
                 local_epochs, optimizer, num_users=100):
        super().__init__(dataset, model, batch_size, learning_rate, num_glob_iters,
                         local_epochs, optimizer, num_users)

        # Initialize the users
        for i in range(num_users):
            user = UserAVG(i, dataset, model, batch_size, learning_rate,
                           local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Finished creating server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            loss_ = 0
            self.send_parameters()
            for user in self.users:
                loss_ += user.train(self.local_epochs) * user.train_samples
            self.aggregate_parameters()
            loss_ /= self.total_train_samples
            loss.append(loss_)
            print(loss_)
        print(loss)
        self.save_model()


class ServerFEDL(Server):
    def __init__(self, dataset, model, batch_size, learning_rate, num_glob_iters,
                 local_epochs, optimizer, num_users=100, eta=0.25):
        super().__init__(dataset, model, batch_size, learning_rate, num_glob_iters,
                         local_epochs, optimizer, num_users)

        # Hyper-learning rate
        self.eta = eta

        for i in range(num_users):
            user = UserFEDL(i, dataset, model, batch_size, learning_rate,
                            local_epochs, optimizer, eta)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        self.send_parameters()
        self.train_loss = torch.zeros(self.num_glob_iters, dtype=torch.float32)
        print("Finished creating server.")

    def send_grads(self, glob_iter=0):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.save_server_grads(grads)

    def train(self):
        for glob_iter in range(self.num_glob_iters):

            # 1: Find \nabla \bar{F} and send it to users
            self.aggregate_grads()
            self.send_grads()

            # 2: Local problem-solving (find w_t^n)
            loss = 0
            for user in self.users:
                loss += user.train(self.local_epochs) * user.train_samples / self.total_train_samples
            print(loss)
            self.train_loss[glob_iter] = loss

            # 3: Find w^t and send it to users (to find \nabla \F_n(w^t))
            self.aggregate_parameters()
            self.send_parameters()

        print(self.train_loss)
        import matplotlib.pyplot as plt
        plt.plot(range(self.num_glob_iters), self.train_loss)
        plt.savefig("josh.png")
        self.save_model()
