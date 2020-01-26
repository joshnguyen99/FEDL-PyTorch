from FedServer import ServerAVG, ServerFEDL
import argparse


def main(dataset, model, batch_size, learning_rate, num_glob_iters,
         local_epochs, optimizer):
    server = ServerFEDL(dataset, model, batch_size, learning_rate,
                        num_glob_iters, local_epochs, optimizer)
    server.train(num_glob_iters)
    server.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "femnist"])
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mclr"])
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_global_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=3)
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
