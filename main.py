import pandas as pd
from NN import Network
import numpy as np


def load_data():
    data = pd.read_csv('cancer1.dt', header=None, sep=" ")
    data.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'target', 'target_1']
    data['weight'] = 1 / data.shape[0]
    data = data.values.tolist()
    return data


def compute_loss(network, data):
    err = 0
    weight_sum = 0
    errors = []
    for test in data:
        result, ref = network.feed_forward(False, test)
        result = np.round(np.squeeze(result))
        weight_sum += ref[1]
        if ref[0] != result:
            err += 1 * ref[1]

            errors.append(True)
        else:
            errors.append(False)
    em = err / weight_sum
    am = (1-em) / em
    am = np.log(am)
    network.weight = am
    return update_data_weights(data, am, errors)


def update_data_weights(data, am, errors):
    for i in range(len(data)):
        if errors[i]:
            data[i][-1] = data[i][-1] * np.exp(am)
    return data


def make_train_data(data, size):
    weight_sum = 0
    threshold_decay = 1
    new_data = []
    for train in data:
        weight_sum += train[-1]
    while len(new_data) != size:
        threshold = np.random.random_sample() / 10
        for train in data:
            if len(new_data) == size:
                break
            if threshold > train[-1]/weight_sum:
                new_data.append(train)
            else:
                continue
        if threshold_decay < 0.5:
            continue
        else:
            threshold_decay -= 0.05
    return new_data


def make_com_predictions(nets, test_data):
    err = 0
    predictions = []
    truth = []
    for test in test_data:
        truth.append(test[-2])
        prediction = 0
        for net in nets:
            res, _ = net.feed_forward(False, test)
            res = res[0] * net.weight
            prediction += res
        predictions.append(prediction)

    for i in range(len(truth)):
        err += np.square(truth[i] - predictions[i]) / 2
    return 1 - err/len(truth)


def start_learn(size_of_com):
    train_data = load_data()
    networks = []
    train_size = int(len(train_data) * 0.8)
    test_size = len(train_data) - train_size
    for _ in range(size_of_com):
        nn = Network(train_data, 1 / size_of_com)
        for _ in range(train_size):
            nn.feed_forward()
        train_data = compute_loss(nn, train_data)
        train_data = make_train_data(train_data, train_size)
        networks.append(nn)

    return make_com_predictions(networks, train_data[:test_size])


def start(size_of_com):
    errors = []
    for i in range(10):
        errors.append(start_learn(size_of_com))
        print('accuracy =', errors[i])

    print('mean =', np.mean(errors))
    print('std =', np.std(errors))


if __name__ == '__main__':
    start(6)
