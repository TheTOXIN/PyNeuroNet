import neiron as n
import data as d
import utill as u
import numpy as np
import sys


epochs = 5000
learning_rate = 0.1
network = n.Neiron(learning_rate=learning_rate)
losses = {'train': [], 'validation': []}

def training():
    for e in range(epochs):
        inputs_ = []
        correct_predictions = []
        for input_stat, correct_predict in d.train:
            network.train(np.array(input_stat), correct_predict)
            inputs_.append(np.array(input_stat))
            correct_predictions.append(np.array(correct_predict))

        train_loss = u.MSE(network.predict(np.array(inputs_).T), np.array(correct_predictions))
        sys.stdout.write(
            "\rProgress: {}, Training loss: {}"
                .format(str(100 * e/float(epochs))[:4], str(train_loss)[:5])
        )