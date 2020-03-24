#!/usr/bin/env python
from FedServer import ServerAVG, ServerFEDL
import argparse


def main(dataset, model, batch_size, learning_rate, num_glob_iters,
         local_epochs, optimizer, eta):
    server = ServerFEDL(dataset, model, batch_size, learning_rate,
                        num_glob_iters, local_epochs, optimizer, 100, eta)
    server.train()
    server.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "femnist"])
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mclr"])
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_global_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--eta", type=float, default=0.3, help="Hyper-learning rate")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Batch size: {}".format(args.batch_size))
    print("h_k       : {}".format(args.learning_rate))
    print("eta       : {}".format(args.eta))
    print("K_g       : {}".format(args.num_global_iters))
    print("K_l       : {}".format(args.local_epochs))
    print("=" * 80)

    main(
        dataset=args.dataset,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer,
        eta=args.eta
    )
