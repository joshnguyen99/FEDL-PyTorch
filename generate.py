import os, json


def main():

    train_path = os.path.join("datafed", "mnist", "userstrain")
    if not os.path.exists(train_path):
            os.makedirs(train_path)
            
    test_path = os.path.join("datafed", "mnist", "userstest")
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    with open(os.path.join("datafed", "train", "mnist_train.json"), "r") as f:
        test = json.load(f)

    for i in range(100):
        data = {}
        data['id'] = test['users'][i]
        data['X'] = test["user_data"][data['id']]['x']
        data['y'] = test["user_data"][data['id']]['y']
        data['num_samples'] = test["num_samples"][i]
        with open(os.path.join(train_path, data['id'] + ".json"), "w") as f:
            json.dump(data, f)

    with open(os.path.join("datafed", "test", "mnist_test.json"), "r") as f_train:
        test = json.load(f_train)

    for i in range(100):
        data = {}
        data['id'] = test['users'][i]
        data['X'] = test["user_data"][data['id']]['x']
        data['y'] = test["user_data"][data['id']]['y']
        data['num_samples'] = test["num_samples"][i]
        with open(os.path.join(test_path, data['id'] + ".json"), "w") as f_test:
            json.dump(data, f_test)


if __name__ == "__main__":
    main()