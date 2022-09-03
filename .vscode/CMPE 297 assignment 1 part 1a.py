#Simple model with tensorflow where we try to learn the equation y= 3x^2 + 4x + 10 by generating synthetic data and using a 1 layer neural network

#function to create a list of random numbers between 1-100
import random
def random_list():
    list = []
    for i in range(100):
        list.append(random.randint(1,100))
    return list

#function to transform the list of numbers given equation given list
def transform_list(list):
    transformed_list = []
    for i in list:
        transformed_list.append(3*i*i + 4*i + 10)
    return transformed_list

#split given data into test and train randomly
def split_data(X,y):
    X_train = X[:int(len(X)*0.8)]
    X_test = X[int(len(X)*0.8):]
    y_train = y[:int(len(X)*0.8)]
    y_test = y[int(len(X)*0.8):]    
    return X_train,X_test,y_train,y_test

#import tesnorflow,numpy and matplotlib, pandas
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#function to build a neural network with 1 layers sequential with relu activation function
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=[1])
    ])
    return model

#function to train the model
def train_net(net, X,y,epochs):
    net.compile(optimizer='adam', loss='mse')
    net.fit(X,y, epochs)
    return net

#function to plot training and test data
def plot_data(X_train,y_train,X_test,y_pred):
    plt.plot(X_train,y_train, 'ro')
    plt.plot(X_test, y_pred, 'b')
    plt.show()
    import msvcrt as m
    m.getch() 
    plt.clf()


#main function
if __name__ == '__main__':
    #create a list of random numbers
    list = random_list()
    #scale the list using min max scaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    list = scaler.fit_transform(np.array(list).reshape(-1,1))

    #transform the list of numbers using the given equation
    transformed_list = transform_list(list)
    scaler = MinMaxScaler()
    transformed_list = scaler.fit_transform(np.array(transformed_list).reshape(-1,1))

    #split the data into train and test
    X_train,X_test,y_train,y_test = split_data(list,transformed_list)
    #build the model
    model = build_model()
    #train the model
    train_net(model,X_train,y_train,epochs=100)

    #predict the test data
    y_pred = model.predict(X_test)

    #plot data
    plot_data(X_train,y_train,X_test,y_pred)

    #wait for key press to exit
