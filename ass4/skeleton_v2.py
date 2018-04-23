import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#standard logistic function to map the inner product w^T * x, and classify
#start
def logistic_z(z):
    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x):
    return logistic_z(np.inner(w,x))

def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic_wx(w,x)<0.5) else 1
#end

#loss function for w in R^2
def L_simple(w):
    return ((logistic_wx(w,[1,0])-1)**2 + logistic_wx(w,[0,1])**2 + (logistic_wx(w,[1,1])-1)**2)
#task 1.1
def plot_L_simple():
    w_1=np.arange(-6,6.1,0.1)
    w_2=np.arange(-6,6.1,0.1)
    w1,w2=np.meshgrid(w_1,w_2)
    loss=np.empty(w1.shape)
    w1_min=10
    w2_min=10
    loss_min=10
    for i in range(len(w1)):
        for j in range(len(w2)):
            coords=[w1[i,j],w2[i,j]]
            loss[i,j]=L_simple(coords)
            if (loss[i,j]<loss_min):
                loss_min=loss[i,j]
                w1_min=w1[i,j]
                w2_min=w2[i,j]

    print("w1_min: ", w1_min)
    print("w2_min: ", w2_min)
    print("loss_min: ", loss_min)

    fig=plt.figure()
    axes=plt.axes(projection='3d')
    axes.contour3D(w1,w2,loss,500)
    axes.set_xlabel('$w_{1}$')
    axes.set_ylabel('$w_{2}$')
    axes.set_zlabel('$L_{Simple}(w)$')
    fig
    plt.show()

#task 1.2
def grad(w):
    L_w1=-((2*np.exp(w[0]))*((6*np.exp(w[0]+w[1]))+(3*np.exp(2*(w[0]+w[1])))+(np.exp(3*(w[0]+w[1])))+(3*np.exp((2*w[0])+w[1]))+(np.exp((3*w[0])+w[1]))+(np.exp(w[1]))+1))/(((np.exp(w[0])+1)**3)*(np.exp(w[0]+w[1])+1)**3)
    L_w2=(2*np.exp(2*w[1])/((np.exp(w[1])+1)**3))-((2*np.exp(w[0]+w[1]))/((np.exp(w[0]+w[1])+1)**3))
    return L_w1, L_w2

#task 1.3
def gradient_descent(learn_rate, niter):
    w_1=0 #evt random
    w_2=0
    for i in range(niter):
        L_w1, L_w2=grad([w_1,w_2])
        w_1=w_1 - (learn_rate*L_w1)
        w_2=w_2 - (learn_rate*L_w2)

    return w_1, w_2

def plot_L_simple_grad_descent(niter):
    learn_rate=0.0001
    learn_rates=[]
    loss_simple=[]
    weights=[]

    while (learn_rate<=100):
        w=gradient_descent(learn_rate,niter)
        loss=L_simple(w)
        weights.append(w)
        loss_simple.append(loss)
        learn_rates.append(learn_rate)
        learn_rate=learn_rate*10

    print('Iterations:', niter)
    for i in range(len(learn_rates)):
        print('Times:', i)
        print('Learn rates:', learn_rates[i])
        print('Weights:', weights[i])
        print('L_simple:', loss_simple[i])
    plt.semilogx(learn_rates,loss_simple, 'ro')
    plt.grid(True)
    plt.xlabel('$\eta$')
    plt.ylabel('$L_{simple}(w)$')
    plt.show()

#task 2.1 & 2.2
def der_Ln(w,x_n,y_n, index):
    sigma=logistic_wx(w,x_n)
    return x[index]*sigma*(1-sigma)*(sigma-y_n)

#x_train = [number_of_samples,number_of_features] = number_of_samples x \in R^number_of_features
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
            update_grad = der_Ln(w,x,y,i) # done ### something needs to be done here
            w[i] = w[i] - (learn_rate*update_grad) # done ### something needs to be done here
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
                update_grad+= der_Ln(w,x_train[n],y_train[n],i)# done # something needs to be done here (-logistic_wx(w,x_train[n])+y_train[n])
            w[i] = w[i] + learn_rate * update_grad/num_n
    return w

def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=10):
    plt.figure()
    #train data
    data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    ax=data.plot(kind='scatter',x='x',y='y',c='lab',cmap=cm.copper,edgecolors='black')

    #train weights
    w=training_method(xtrain,ytrain,learn_rate,niter)
    error=[]
    y_est=[]
    for i in xrange(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm,edgecolors='black')
    print "error=",np.mean(error)
    return w

def main():
    plot_L_simple_grad_descent(10000)

    #small nonsep
    x_train_small_nonsep=np.loadtxt("data_small_nonsep_train.csv", delimiter='\t', usecols=(0,1))
    y_train_small_nonsep=np.loadtxt("data_small_nonsep_train.csv", delimiter='\t', usecols=(2))
    x_test_small_nonsep=np.loadtxt("data_small_nonsep_test.csv", delimiter='\t', usecols=(0,1))
    y_test_small_nonsep=np.loadtxt("data_small_nonsep_test.csv", delimiter='\t', usecols=(2))

    #small sep
    x_train_small_sep=np.loadtxt("data_small_seperable_train.csv", delimiter='\t', usecols=(0,1))
    y_train_small_sep=np.loadtxt("data_small_seperable_train.csv", delimiter='\t', usecols=(2))
    x_test_small_sep=np.loadtxt("data_small_seperable_test.csv", delimiter='\t', usecols=(0,1))
    y_test_small_sep=np.loadtxt("data_small_seperable_test.csv", delimiter='\t', usecols=(2))

    #big nonsep
    x_train_big_nonsep=np.loadtxt("data_big_nonsep_train.csv", delimiter='\t', usecols=(0,1))
    y_train_big_nonsep=np.loadtxt("data_big_nonsep_train.csv", delimiter='\t', usecols=(2))
    x_test_big_nonsep=np.loadtxt("data_big_nonsep_test.csv", delimiter='\t', usecols=(0,1))
    y_test_big_nonsep=np.loadtxt("data_big_nonsep_test.csv", delimiter='\t', usecols=(2))

    #big sep
    x_train_big_sep=np.loadtxt("data_big_seperable_train.csv", delimiter='\t', usecols=(0,1))
    y_train_big_sep=np.loadtxt("data_big_seperable_train.csv", delimiter='\t', usecols=(2))
    x_test_big_sep=np.loadtxt("data_big_seperable_test.csv", delimiter='\t', usecols=(0,1))
    y_test_big_sep=np.loadtxt("data_big_seperable_test.csv", delimiter='\t', usecols=(2))

main()