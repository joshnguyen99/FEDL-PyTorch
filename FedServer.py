import torch
import os
from tqdm import tqdm
from FedUser import User
from FedModel import Net
import argparse


class Server:
    def __init__(self, dataset, model, batch_size, learning_rate,
                 num_glob_iters, local_epochs, optimizer, num_users=100):
        self.dataset, self.num_global_iters, self.local_epochs, self.batch_size = \
            dataset, num_glob_iters, local_epochs, batch_size
        self.learning_rate = learning_rate

        self.users = []
        self.total_train_samples = 0
        for i in range(num_users):
            user = User(i, dataset, model, batch_size, learning_rate,
                        local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        if model == "cnn":
            if self.model_exists():
                self.load_model()
            else:
                self.model = Net()
        print("Finished creating server.")

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model.parameters())

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param = torch.zeros_like(param)
        for user in self.users:
            self.add_parameters(user, user.train_samples / self.total_train_samples)

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        for user in self.users:
            user.set_grads(grads)
            # user.set_parameters(self.model.parameters())

    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param = torch.zeros_like(param)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameter()):
            param.grad.data = param.grad.data + user_grad[idx].clone() * ratio

    def add_parameters(self, user, ratio):
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param = server_param + user_param.clone() * ratio

    def train(self, num_glob_iters=1):
        for glob_iter in tqdm(range(1, num_glob_iters + 1), desc="Global Iteration"):
            self.send_parameters()
            for user in tqdm(self.users, desc="User"):
                user.train(3)
            self.aggregate_parameters()
        self.save_model()

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


def main(dataset, model, batch_size, learning_rate, num_glob_iters,
         local_epochs, optimizer):
    server = Server(dataset, model, batch_size, learning_rate,
                    num_glob_iters, local_epochs, optimizer)
    server.train(3)
    server.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "femnist"])
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mclr"])
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_global_iters", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--optimizer", type=str, default="SGD")
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer
    )
