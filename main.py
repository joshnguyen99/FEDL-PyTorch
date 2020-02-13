import argparse


def main(dataset, model, batch_size, learning_rate, num_glob_iters,
         local_epochs, optimizer, eta):
    
    if dataset == "mnist":
        from flearn.mnist.FedServer import ServerAVG, ServerFEDL
    elif dataset == "synthetic_logistic_regression":
        from flearn.synthetic.logistic_regression.FedServer import ServerAVG, ServerFEDL
    elif dataset == "synthetic_linear_regression":
        from flearn.synthetic.linear_regression.FedServer import ServerAVG, ServerFEDL
    
    server = ServerFEDL(dataset, model, batch_size, learning_rate,
                        num_glob_iters, local_epochs, optimizer, 100, eta)
    server.train()
    server.test()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "synthetic_logistic_regression",
                                 "synthetic_linear_regression"])

    parser.add_argument("--model", type=str, default="FEDL", choices=["FEDL", "FedAvg"])

    parser.add_argument("--batch_size", type=int, default=100)

    parser.add_argument("--learning_rate", type=float, default=0.01)

    parser.add_argument("--num_global_iters", type=int, default=200)

    parser.add_argument("--local_epochs", type=int, default=10)

    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "FEDLOptimizer"])

    parser.add_argument("--eta", type=float, default=0.5, help="Hyper-learning rate")

    args = parser.parse_args()

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
