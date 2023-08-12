"""
Simple Linear Regression Implementation Using Tensorflow
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# Tensorflow implementation
class Model:
    def __init__(self):
        self.weight = tf.Variable(10.0)
        self.bias = tf.Variable(10.0)

    def __call__(self, x):
        return (self.weight * x) + self.bias


def calculate_loss(y_actual, y_predicted):
    return tf.reduce_mean(tf.square(y_actual - y_predicted))  # Its taking square root of the average


def train(model, x, y, learning_rate):
    # https://www.tensorflow.org/guide/autodiff#:~:text=to%20compute%20gradients.-,Gradient%20tapes,GradientTape%20onto%20a%20%22tape%22.
    with tf.GradientTape() as gt:
        y_predicted = model(x)
        loss = calculate_loss(y, y_predicted)

    new_weight, new_bias = gt.gradient(loss, [model.weight, model.bias])
    model.weight.assign_sub(new_weight * learning_rate)
    model.bias.assign_sub(new_bias * learning_rate)


if __name__ == '__main__':
    # TODO: Lets create some synthetic datasets
    m = 2
    b = 0.5
    x = np.linspace(0, 4, 100)  # Input variable

    y = m * x + b + np.random.randn(*x.shape) + 0.25
    # Here y is the output variable which we are generating with some randomness
    # np.random.randn(*x.shape) + 0.25 Add randomness into the data
    plt.scatter(x, y)
    # plt.show()  # TODO: Uncomment this if want to see that data on chart

    model = Model()
    epochs = 100
    learning_rate = 0.15

    for epoch in range(epochs):
        y_predicted = model(x)
        loss = calculate_loss(y, y_predicted)
        print(f"Epoch: {epoch}, Loss: {loss.numpy()}")
        train(model, x, y, learning_rate)

    print(model.weight.numpy())
    print(model.bias.numpy())

    new_x = np.linspace(0, 4, 100)  # Input variable
    new_y = model.weight.numpy() * new_x + model.bias.numpy()
    plt.scatter(new_x, new_y)
    plt.scatter(x, y)
    plt.show()
