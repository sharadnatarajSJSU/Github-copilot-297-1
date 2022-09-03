#we are going to do the same with pytorch

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

#import pytorch,matplotlib,numpy
import torch
import matplotlib.pyplot as plt
import numpy as np


#function to build a neural network with 1 layers sequential with relu activation function
def build_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    return model

#function to train the model
def train_net(net, X,y,epochs):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epochs):
        y_pred = net(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
    #transform the list
    transformed_list = transform_list(list)
    scaler = MinMaxScaler()
    transformed_list = scaler.fit_transform(np.array(transformed_list).reshape(-1,1))
    #split the data into train and test
    X_train,X_test,y_train,y_test = split_data(list,transformed_list)
    #convert the data into tensors
    X_train_t = torch.from_numpy(X_train).float()
    X_test_t = torch.from_numpy(X_test).float()
    y_train_t = torch.from_numpy(y_train).float()
    y_test_t = torch.from_numpy(y_test).float()
    #build the model
    net = build_model()
    #train the model
    net = train_net(net, X_train_t,y_train_t,100)
    #predict the test data
    y_pred = net(X_test_t)
    y_pred = y_pred.detach().numpy() 
    #plot the data
    plot_data(X_train,y_train,X_test,y_pred)

