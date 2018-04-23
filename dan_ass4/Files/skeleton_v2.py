import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits import mplot3d
import time

xrange = range

def logistic_z(z):
    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x):
    return logistic_z(np.inner(w,x))

###################################################################################################################

#equation 2 in the assignment
def L_simple(w):
    return (logistic_wx(w, [1, 0]) - 1)**2 + (logistic_wx(w, [0, 1]))**2 + (logistic_wx(w, [1, 1]) - 1) ** 2

#plotting the loss function for task 1
def plot_L_simple():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    X,Y = np.meshgrid(x,y)
    #Create space for the matrix we want to store data for L_simple in. Since we go from -6 to 6 with 0.1 step length,
    # we will need a 120x120 matrix
    Z = np.zeros(X.shape)

    #Store L_simple values inside the matrix Z
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            Z[i, j] = L_simple([X[i, j], Y[i, j]])

    #Plotting
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    print("Z-verdien:")
    print(Z[4,-2])
    ax.contour3D(X, Y, Z, 100, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.view_init(60, 35)
    fig
    plt.show()

#outputs the gradient. based on the math done in 1b
def gradient(w):
    w1 = w[0]
    w2 = w[1]
    dw1 = -2*np.exp(w1) * ( 6*np.exp(w1+w2) + 3*np.exp(2*(w1+w2)) + np.exp(3*(w1+w2)) + 3*np.exp(2*w1+w2) + np.exp(3*w1+w2) + np.exp(w2) + 1)/ ((np.exp(w1)+1)**3 * (np.exp(w1+w2) + 1)**3)
    dw2 = 2*np.exp(2*w2) / (np.exp(w2)+1)**3 - 2*np.exp(w1+w2)/(np.exp(w1+w2)+1)**3
    return dw1, dw2

#finds the minimum point and returns the point. input a random weight vector
def gradient_descent(learn_rate=100, niter=1000):

    '''
    #generate random starting weights between -6 and 6 for both weights
    w1 = np.random.randint(-6,6)
    w2 = np.random.randint(-6,6)
    '''
    w1 = 0
    w2 = 0

    iteration = 0
    #go through niter iterations
    while iteration != niter:
        dw1, dw2 = gradient([w1, w2])

        #Update
        w1 = w1 - learn_rate * dw1
        w2 = w2 - learn_rate * dw2
        iteration=iteration+1
    return w1, w2

#returns results for gradient descent from a given list of different etas
def gradient_descent_etas(niter=1000):
    etas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    loss_values = []

    for eta in etas:
        w1, w2 = gradient_descent(eta, niter)
        print("eta:")
        print(eta)
        print("loss value: ")
        w = [w1, w2]
        print(L_simple(w))
        loss_values.append(L_simple(w))
        print("\n")

    plt.plot(etas, loss_values, 'ro')
    plt.xlabel("$\eta$")
    plt.ylabel("$L_{simple}$")
    plt.show()

#answer to task 2a. i is the index for which weight we are analyzing. w is the weight. x is the example features.
#y is the example target (which class we want to be in)
def gradient_Ln(w, x, y, i):
    x_i = x[i]
    val = (logistic_wx(w,x) - y) * x_i * logistic_wx(w,x) * (1-logistic_wx(w,x))
    return val

###################################################################################################################

def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic_wx(w,x)<0.5) else 1

def stochast_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in xrange(niter):
        if(len(index_lst)==0):
            index_lst=random.sample(xrange(num_n), k=num_n)
        xy_index = index_lst.pop()
        x=x_train[xy_index,:]
        y=y_train[xy_index]
        for i in xrange(dim):
            update_grad = gradient_Ln(w, x, y, i) ### something needs to be done here. DONE. Point i in stochastic algo in assignment
            w[i] = w[i] - learn_rate * update_grad ### something needs to be done here. DONE. Point ii in stochastic algo in assignment
    return w

def batch_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in xrange(niter):
        for i in xrange(dim):
            update_grad=0.0
            for n in xrange(num_n):
                update_grad += gradient_Ln(w, x_train[n], y_train[n], i)# something needs to be done here. DONE. Point A in batch algo in assignment
            w[i] = w[i] - learn_rate * update_grad/num_n
    return w

def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=10):
    plt.figure()
    #train data
    data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    ax=data.plot(kind='scatter',x='x',y='y',c='lab',cmap=cm.copper,edgecolors='black')

    #train weights
    startTime = time.time()
    w=training_method(xtrain,ytrain,learn_rate,niter)
    endTime = time.time()
    totalDuration = endTime - startTime
    print("Total duration: ", round(totalDuration,10))
    error=[]
    y_est=[]
    for i in xrange(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm,edgecolors='black')
    print("error=",np.mean(error))

    plt.show()

    return w


if __name__ == '__main__':
    #print(gradient([0,0]))
    #print(gradient_descent())
    gradient_descent_etas()
    #plot_L_simple()

    # print("Importing")
    #
    # x_train = np.loadtxt("data/data_small_separable_train.csv", delimiter="\t", usecols=(0,1))
    # y_train = np.loadtxt("data/data_small_separable_train.csv", delimiter="\t", usecols=(2,))
    #
    # x_test = np.loadtxt("data/data_small_separable_test.csv", delimiter="\t", usecols=(0,1))
    # y_test = np.loadtxt("data/data_small_separable_test.csv", delimiter="\t", usecols=(2,))
    #
    # print("Import successful")
    #
    # train_and_plot(x_train, y_train, x_test, y_test, stochast_train_w, learn_rate=0.1,niter=500)
    #train_and_plot(x_train, y_train, x_test, y_test, batch_train_w, learn_rate=0.1, niter=100)


'''
Sprsml
Hva er fargekodingen i scatterplottet
Hvordan 'keep track of training time
'''
