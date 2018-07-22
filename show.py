import numpy as np
import data as d
import train as t


def showResult():
    for input_stat, correct_predict in d.train:
        print("For input: {} the prediction is: {} - {}, expected: {}".format(
            str(input_stat),
            str(t.network.predict(np.array(input_stat)) > 0.5),
            str(t.network.predict(np.array(input_stat))),
            str(correct_predict == 1)
        ))
