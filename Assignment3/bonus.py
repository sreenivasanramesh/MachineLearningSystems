'''
4 or 9
For this bonus task, I am going to use logistic regression to
predict between two classes. I am using the MNIST dataset, and
predicting between class 4 and 9, cause those two classes had the
highest misclassification last time. The two classes can be
changed in the main function.

Note: sklearn is required to be installed to normalize the data for better accuracy


author: Sreenivasan Ramesh - sreenivasan.ramesh@asu.edu
'''


import random
import os
import warnings
import numpy as np
from sklearn import preprocessing

import autodiff as ad
import utils

warnings.filterwarnings("ignore")



def fetch_data(class_1, class_2):
    '''reduces MNIST data to a two class classification problem'''
    mnist_folder = 'data/mnist'
    if not os.path.isdir(mnist_folder):
        os.mkdir('data')
        os.mkdir(mnist_folder)
    utils.download_mnist(mnist_folder)
    train, _, test = utils.read_mnist(mnist_folder, flatten=True)

    train_images = np.array(train[0])
    train_images = preprocessing.scale(np.array(train_images), axis=1)
    train_labels = np.array([np.where(x == 1)[0][0] for x in train[1]])
    test_images = np.array(test[0])
    test_images = preprocessing.scale(np.array(test_images), axis=1)
    test_labels = np.array([np.where(x == 1)[0][0] for x in test[1]])

    train_data = ((train_labels == class_1) + (train_labels == class_2))
    x_train = train_images[train_data]
    y_train = train_labels[train_data]
    test_data = ((test_labels == class_1) + (test_labels == class_2))
    x_test = test_images[test_data]
    y_test = test_labels[test_data]

    return x_train, y_train, x_test, y_test



def get_model_params(x_train, y_train, class_1, class_2):
    '''returns the weights after performing gradient descdent'''
    learning_rate = 0.01
    batch_size = 8

    x = ad.Variable(name='x')
    w = ad.Variable(name='w')
    y = ad.Variable(name='y')

    logistic_regression = 1 / (1 + ad.exp_op(0 - ad.matmul_op(w, x)))
    cross_entropy = -1 * y * ad.log_op(logistic_regression) - (1 - y) * ad.log_op(1 - logistic_regression)

    gradients = ad.gradients(cross_entropy, [w])[0]
    executor = ad.Executor([cross_entropy, gradients])
    weights = np.random.rand(1, np.shape(x_train)[1]) / 1000.0

    #batch = 0
    #previous_loss = 0
    for i in range(5000):
        grad = np.zeros((1, np.shape(x_train)[1]))
        loss = 0

        #go ramdomly over examples in each batch
        for _ in range(batch_size):
            t = random.choice(range(np.shape(x_train)[0]))
            x_flat = x_train[t].reshape((np.shape(x_train)[1], 1))
            y_label = 0 if y_train[t] == class_1 else 1

            loss_delta, grad_delta = executor.run(feed_dict={x : x_flat, w : weights, y : y_label})
            grad += grad_delta
            loss += loss_delta
        weights = weights - (learning_rate * grad / batch_size)
        if i % 1000 == 0:
            print("loss = {:.3f} loss_delta = {:.3f}".format(loss[0][0], loss_delta[0][0]))

    return weights



def predict(x_test, weights, class_1, class_2):
    '''given model weights, predict the class'''
    sig = 1 / (1 + np.exp(-1 * np.dot(weights, x_test)))
    if  sig < 0.5:
        return class_1
    return class_2



def main():
    '''
    main function - gets the two class dataset, trains a model
    which uses logistic regression, and tests on test set to
    get the accuracy
    '''

    #lets reuce the MNIST data to only my two classes
    class_1 = 4
    class_2 = 9
    x_train, y_train, x_test, y_test = fetch_data(class_1, class_2)
    weights = get_model_params(x_train, y_train, class_1, class_2)

    #get accuracy
    correct_predictions = 0
    for i in range(np.shape(x_test)[0]):
        y_pred = predict(x_test[i], weights, class_1, class_2)
        if y_pred == y_test[i]:
            correct_predictions += 1

    print("Accuracy (4 vs 9) = {:.2f}".format(correct_predictions / len(y_test) * 100))


main()
