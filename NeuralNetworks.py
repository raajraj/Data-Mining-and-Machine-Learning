#!/usr/bin/env python
# coding: utf-8

# 
# # Problem Set 3
# # Due Sunday, November 22, 2020
# 
# <div style="color: #000000;background-color: #FFEEFF">
# In this problem set, you are to implement a three-layer (3 layers of weights, 2 hidden layers of units) neural network for binary classification.  All non-linearities are to be sigmoids.
# 
# Details are given below.  *Please read the **entire** notebook carefully before proceeding.*
# 
# You need to both fill in the necesary code, **and** answer the question at the bottom.</div>

# ### Enter your information below:
# 
# <div style="color: #000000;background-color: #FFCCCC">
#     Your Name (submitter): Raajitha Rajkumar <br>
# Your student ID (submitter): 862015848
# </div>

# <div style="color: #000000;background-color: #FFCCFF">
#     Total Grade: /30<br>
#     Late Days on this assigment: 1 <br>
#     Total Late Days so far: 4 <br>
#     </div>

# In[1]:


# Below are the only imports that are necessary (or allowed)
import numpy as np
import h5py 
import matplotlib.pyplot as plt
import time
from IPython import display


# ## Data
# 
# <div style="color: #000000;background-color: #FFEEFF">
# We will be using a USPS digit dataset (provided in the file uspsall73.mat).
# It has 16-by-16 grayscale images of each of the 10 different hand-written digits
# However, we will load only two of the digits to use as the two classes in
# binary classification
# </div>

# In[2]:


# function to load two of the 10 classes (c1 is for Y=+1 and c2 is for Y=0)
# Note that for neural networks, we will be using Y={+1,0} instead of Y={+1,-1}
def loaddigitdata(c1,c2,m):
    f = h5py.File('uspsall73.mat','r') 
    data = f.get('data') 
    data = np.array(data).astype(float)
    X = np.concatenate((data[c1,:,:],data[c2,:,:]))
    Y = np.concatenate((np.zeros((data.shape[1])),np.ones((data.shape[1]))))
    
    rndstate = np.random.get_state() # going to set the "random" shuffle random seed
    np.random.seed(132857) # setting seed so that dataset is consistent
    p = np.random.permutation(X.shape[0])
    X = X[p] # this and next line make copies, but that's okay given how small our dataset is
    Y = Y[p]
    np.random.set_state(rndstate) # reset seed
    
    trainX = X[0:m,:] # use the first m (after shuffling) for training
    trainY = Y[0:m,np.newaxis]
    validX = X[m:,:] # use the rest for validation
    validY = Y[m:,np.newaxis]
    return (trainX,trainY,validX,validY)

# In case you care (not necessary for the assignment)
def drawexample(x,ax=None): # takes an x *vector* and draws the image it encodes
    if ax is None:
        plt.imshow(np.reshape(x,(16,16)).T,cmap='gray')
    else:
        ax.imshow(np.reshape(x,(16,16)).T,cmap='gray')


# In[3]:


# load the data, to differentiate between 7s and 9s
# we will use on 1100 examples for training (50% of the data) and the other half for validation
(trainX,trainY,validX,validY) = loaddigitdata(6,8,1100)
means = trainX.mean(axis=0)
stddevs = trainX.std(axis=0)
stddevs[stddevs<1e-6] = 1.0
trainX = (trainX-means)/stddevs # z-score normalization
validX = (validX-means)/stddevs # apply same transformation to validation set

# Convert this cell to a code cell if you wish to see each of the examples, plotted
# (completely not necessary for the problem set)
f = plt.figure()
f.set_size_inches(8,8)

ax = f.add_subplot(111)
plt.ion()
f.canvas.draw()
for exi in range(trainX.shape[0]):
    display.clear_output(wait=True)
    drawexample(trainX[exi,:])
    digitid = (9 if trainY[exi]>0 else 7)
    ax.set_title('y = '+str(int(trainY[exi]))+" ["+str(digitid)+"]")
    display.display(f)
    #time.sleep(0.1)
# ## WRITE `nneval` and `trainneuralnet` [25 points]
# 
# <div style="color: #000000;background-color: #FFFFEE">
# This is the main portion of the assignment
# 
# Note that the $Y$ values are +1 and 0 (not +1 and -1).  This is as in class for neural networks and works better with a sigmoid output.
# 
# You need to write the two functions below (plus any more you would like to add to help): `nneval` and `trainneuralnet`.  The first takes an array/matrix of X vectors and the weights from a neural network and returns a vector of predicted Y values (should be numbers between 0 and 1 -- the probability of class +1, for each of the examples).  The second takes a data set (Xs and Ys), the number of hidden units, and the lambda value (for regularization), and returns the weights.  W1 are the weights from the input to the hidden and W2 are the weights from the hidden to the output.
# 
# A few notes:
# - **Starting Weights**: The code supplied randomly selects the weights near zero.  https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79 has a reasonable explanation of why we are doing it this way.  But for the purposes of the assignment, you can just accept this is a good way to initialize neural network weights.
# - **Offset Terms**: Each layer should have an "offset" or "intercept" unit (to supply a 1 to the next layer), except the output layer.
# - **Batch Updates**: For a problem this small, use batch updates.  That is, the step is based on the sum of the gradients for each data point in the training set.
# - **Step Size**: http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms describes a number of methods to adaptively control $\eta$ for fast convergence.  You don't need to understand any of them; however, without them, convergence to good solutions on this problem can be quite slow.  Therefore, *use RMSprop*: the code below has a simple version of RMSprop that is sufficient for this assignment.  You need to supply the code that calculates `sumofgrad2` which should be the sum of the square of each element of the gradient (the squared length of the gradient).  (for debugging, feel free to use a constant $\eta$). 
# - **Stopping Criterion**: To determine when to stop, check the loss function every 10 iterations.  If it has not improved by at least $10^{-6}$ over those 10 iterations, stop.
# - **Regularization**: You should penalize (from the regularization) all of the weights, even those coming out of offset units.  While it makes sense sometimes not to penalize the ones for the constant $1$ units, you'll find this easier if you just penalize them all.
# 
# Tips that might help:
# - Display the loss function's value every 10 iterations (or so).  It should be getting smaller.  If not, your gradient is not pointing in the right direction.
# - The smaller $\lambda$ is and the more units, the more difficult (longer) the optimization will be.
# - Write a function to do forward propagation and one to do backward propagation.  Write a function to evaluate the loss function.  In general, break things up to keep things straight.
# - Processing the entire batch at once is more efficient in numpy.  Use numpy broadcasting to avoid loops where possible.
# </div>

# <div style="color: #000000;background-color: #FFCCFF">
#     Grade: /25<br>
#     Comments:
#     </div>

# In[4]:


### FEEL FREE TO ADD HELPER FUNCTIONS

def forward_propagation(x, w, p):
    if(p < 3):
        z = np.matmul(x,w.T)
        a = sigmoid_all(z)
        a = np.insert(a, 0, 1.00)
    if(p == 3):
        z = np.dot(x,w[0])
        a = sigmoid(z)
        
    return a

def backwards_propagation(delta, w, a, p):
    alen = len(a)
    b = np.empty(alen)
    
    if(p == 1):
        d = np.multiply(delta, w.T)
    else:
        d = np.matmul(delta, w)
        
    for i in range(alen):
        b[i] = 1 - a[i]   
    gz = np.multiply(a, b)
    
    delta = np.multiply(d.T, gz)
    return delta

def gradient(delta, x, g):
    dlen = len(delta)
    (m, n) = g.shape

    gradient = np.zeros(shape = (m, n))
    for i in range(m):
        for j in range(n):
            gradient[i][j] = delta[i]*x[j]
    
    return gradient  
        
def check_loss(m, lam, W1, W2, W3, Y, f, iteration):
    loss = 0
    regularization = (np.sum(W1**2.00) + np.sum(W2**2.00) + np.sum(W3**2.00))*lam/m
    for i in range(10):
        loss = loss + ((Y[(iteration*10) + i] - f[i])**2)
    
    loss = loss/m + regularization                 
    print(loss)
    
    return loss
    
def sigmoid(value):
    natural_log_powerval = np.exp(-value)
    result = 1/(1 + natural_log_powerval)
    return result 

def sigmoid_all(value_all):
    vallen = len(value_all)
    result_all = np.empty(vallen)
    for i in range(vallen):
        natural_log_powerval_all = np.exp(-value_all[i])
        result_all[i] = 1/(1 + natural_log_powerval_all)
    return result_all


# Wts is whatever object you return from trainneuralnet
def nneval(X, Wts):
    # YOUR CODE HERE
    # Should return a vector of values (btwn 0 and 1) for each of the rows of X
    X = np.insert(X, 0, 1.00, axis = 1)
    (m, n) = X.shape
    y = np.empty(m)
    
    for i in range(m):
        a1 = forward_propagation(X[i], Wts[0], 1)
        a2 = forward_propagation(a1, Wts[1], 2)
        a3 = forward_propagation(a2, Wts[2], 3)
        
        # print(a3)
        if(((a3*(10**40)) % 10) > 0.00):
            y[i] = 1.00
        else:
            y[i] = 0.00
    
    print(y)
    
    
    return y

# Your functions need only work for neural networks of exactly 3 layers of weights
# This training function has a single scalar parameter, nhid, to indicate the number of
# hidden units.  This is the number in the first hidden layer.  The second hidden layer will have 1/2 this number
# You can use "printinfo" to control whether to print out debugging info (or you can just ignore it)
def trainneuralnet(X,Y,nhid,lam,printinfo=False):
    # The number of examples (m) and number of input dimensions (n) of the input data
    (m,n) = X.shape

    # This is the code that initializes the weigth matrices:
    # W1 is nhid by n+1 (to map from the input, plus a constant term, to the first hidden layer)
    # W2 is nhid/2 by nhid+1 (to map from the first hidden layer of units to the second)
    # W3 is nhid/2+1 by 1 (to map from the second layers of hidden units to the output)
    W1 = (np.random.rand(nhid,n+1)*2-1)*np.sqrt(6.0/(n+nhid+1)) # weights to each hidden unit from the inputs (plus the added offset unit)
    W2 = (np.random.rand(nhid//2,nhid+1)*2-1)*np.sqrt(6.0/(nhid+nhid/2+1))
    W3 = (np.random.rand(1,nhid//2+1)*2-1)*np.sqrt(6.0/(nhid+2)) # weights to the single output unit from the hidden units (plus the offset unit)
    W1[:,0] = 0 # good initializations for the constant terms
    W2[:,0] = -nhid/2.0
    W3[:,0] = -nhid/4.0
    Wts = [W1,W2,W3] # I put them together in a list, but you can use any structure to keep them together and return them in the end
    Eg2=1
    
    checker = 10 # checker for checking loss every 10 iterations
    f = [] # f array for f values
    X = np.insert(X, 0, 1.00, axis = 1) # offset beginning values to 1
    prev_loss = 0 # set current previous loss to 0
    iteration = 0 # checks which loss iteration we are on of every 10
    
    # Your loop here:
    for i in range(nhid):
        
        # Forward propagation
        a1 = forward_propagation(X[i], W1, 1)
        a2 = forward_propagation(a1, W2, 2)
        a3 = forward_propagation(a2, W3, 3)

        # Backwards propagation
        delta_output = a3 - Y[i]
        delta2 = backwards_propagation(delta_output, W3, a2, 1)
        delta2 = np.delete(delta2, 0)
        delta1 = backwards_propagation(delta2, W2, a1, 2)
        delta1 = np.delete(delta1, 0)
        f.append(delta_output)
        
        # Gradient operation
        gradient1 = gradient(delta1, X[i], W1)
        gradient2 = gradient(delta2, a1, W2)
        gradient3 = gradient(delta_output, a2, W3)
        
        # in the loop, after calculating the gradient, but before making a step
        # calculate the sum of the squares of all of the gradient values
        # and store it in sumofgrad2
        # then execute this code to get the step size, eta:
        
        sumofgrad2 = np.sum(gradient1**2) + np.sum(gradient2**2) + np.sum(gradient3**2)
        Eg2 = 0.9*Eg2 + 0.1*sumofgrad2
        eta = 0.01/(np.sqrt((1e-10+Eg2)))
        
        # Update new weights
        W1 = W1 - (eta * gradient1) - ((eta * 2.00 * lam / m) * W1)
        W2 = W2 - (eta * gradient2) - ((eta * 2.00 * lam / m) * W2)
        W3 = W3 - (eta * gradient3) - ((eta * 2.00 * lam / m) * W3)
        
        # checks if 10 iterations has passed, and checks loss if so
        if(checker == 0):
            checker = 10 # resets checker
            iteration = iteration + 1 # sets iteration value
            loss = check_loss(m, lam, W1, W2, W3, Y, f, iteration)
            if(loss - prev_loss < 10**-6): 
                Wts = [W1,W2,W3] # I put them together in a list, but you can use any structure to keep them together and return them in the end
                return Wts 
            prev_loss = loss
            f.clear()
        else:
            checker = checker - 1
            
    # when done, return your weights
    Wts = [W1,W2,W3] # I put them together in a list, but you can use any structure to keep them together and return them in the end
    return Wts 


# In[4]:


### FEEL FREE TO ADD HELPER FUNCTIONS

def addones(Z):
    return np.hstack((np.ones((Z.shape[0],1)),Z))

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def mynneval(X,Wts):
    A = addones(X)
    Zs = [X]
    As = [A]
    for i in range(len(Wts)):
        Z = A@Wts[i].T
        Zs.append(Z)
        temp = sigmoid(Z)
        A = addones(sigmoid(Z))
        As.append(A)
    return (Zs,As)

def nneval(X,Wts):
    return mynneval(X,Wts)[1][-1][:,1:]

def nngrad(X,Y,Wts,lam):
    m = X.shape[0]
    Zs,As = mynneval(X,Wts)
    Deltas = [As[-1][:,1:]-Y]
    for i in range(len(Wts)-1,-1,-1):
        D = Deltas[0]@Wts[i]
        Delta = D*As[i]*(1-As[i])
        Deltas.insert(0,Delta[:,1:])
    Gs = []
    for i in range(len(Wts)):
        Gs.append((Deltas[i+1].T@As[i] + 2*lam*Wts[i])/m)
    return (Gs,As[-1])

def nnloss(ypred,Y,lam,Wts):
    m = Y.shape[0]
    w2 = 0
    for w in Wts:
        w2 += (w*w).sum()
        
    return (-np.log(ypred[Y>0.5]+1e-10).sum())/m - np.log(1.-ypred[Y<=0.5]+1e-10).sum()/m + lam/m*w2

# Your functions need only work for neural networks of exactly 3 layers of weights
# This training function has a single scalar parameter, nhid, to indicate the number of
# hidden units.  This is the number in the first hidden layer.  The second hidden layer will have 1/2 this number
def trainneuralnet(X,Y,nhid,lam,printinfo=False):
    # The number of examples (m) and number of input dimensions (n) of the input data
    (m,n) = X.shape
    
    # This is the code that initializes the weigth matrics:
    # W1 is nhid by n+1 (to map from the input, plus a constant term, to the first hidden layer)
    # W2 is nhid/2 by nhid+1 (to map from the first hidden layer of units to the second)
    # W3 is nhid/2+1 by 1 (to map from the second layers of hidden units to the output)
    W1 = (np.random.rand(nhid,n+1)*2-1)*np.sqrt(6.0/(n+nhid+1)) # weights to each hidden unit from the inputs (plus the added offset unit)
    W2 = (np.random.rand(nhid//2,nhid+1)*2-1)*np.sqrt(6.0/(nhid+nhid/2+1))
    W3 = (np.random.rand(1,nhid//2+1)*2-1)*np.sqrt(6.0/(nhid+2)) # weights to the single output unit from the hidden units (plus the offset unit)
    W1[:,0] = 0 # good initializations for the constant terms
    W2[:,0] = -nhid/2.0
    W3[:,0] = -nhid/4.0
    
    Wts = [W1,W2,W3] # I put them together in a list, but you can use a tuple too
    
    Eg2=1
    j = 1
    delloss = 1
    oldloss = np.Infinity
    while delloss>1e-6:
        for i in range(10):
            (Grad,ypred) = nngrad(X,Y,Wts,lam)
            
            sumofgrad2 = 0
            for G in Grad:
                sumofgrad2 += (G*G).sum()
            Eg2 = 0.9*Eg2 + 0.1*sumofgrad2
            eta = 0.01/(np.sqrt((1e-10+Eg2)))
            
            for i in range(len(Wts)):
                Wts[i] -= eta*Grad[i]
                
        ypred = nneval(X,Wts)
        newloss = nnloss(ypred,Y,lam,Wts)
        delloss = oldloss-newloss
        
        oldloss = newloss
        j = j+1
        #if j%10000 == 0:
        #    print ("i = %d, nhid = %d, lam = %lf, loss = %lf, eta = %lf, gradsum2 = %lf" % (j,nhid,lam,newloss,eta,sumofgrad2))
    
    if (printinfo):
        print(j)
        print(oldloss)
        
    return Wts 


# In[5]:


get_ipython().run_cell_magic('time', '', '# Use this cell (or others you add) to check your network\n# I would debug on simple examples you create yourself (trying to understand what happens with\n#  the full 256-dimensional data is hard)\n\n#an example of training on the USPS data with 32/16 hidden units and lambda=0.1, takes about 9800 iterations and about 50 seconds for the solutions\nWts = trainneuralnet(trainX,trainY, 100, 10000 ,True)\ny = nneval(trainX, Wts)\nprint(trainY)')


# ## Performance plot
# <div class="alert alert-info">
# The code below will plot your algorithm's error rate on this data set for various regularization strengths and numbers of hidden units.
# 
# Make sure your code works for this plot.
# 
# My code runs in about 12 minutes (to produce the full plot below)
# </div>

# In[122]:


get_ipython().run_cell_magic('time', '', "# This code is given.  Do not modify.\ndef setupfig():\n    f = plt.figure()\n    f.set_size_inches(8,8)\n    ax = f.add_subplot(111)\n    plt.ion()\n    f.canvas.draw()\n    return (f,ax)\n\ndef plotit(lams,nhiddens,erates,f,ax):\n    ax.clear()\n    for i in range(nhiddens.shape[0]):\n        ax.plot(lams,erates[:,i],'*-')\n    ax.set_yscale('log',subs=[1,2,3,4,5,6,7,8,9])\n    ax.set_yticks([0.1,0.01])\n    ax.set_xscale('log')\n    f.canvas.draw()\n    ax.set_xlabel('lambda')\n    ax.set_ylabel('validation error rate')\n    ax.legend([(('# hidden units = '+str(x)) if x>0 else 'logistic regression') for x in nhiddens])\n    display.display(f)\n    display.clear_output(wait=True)\n    \ndef errorrate(Y,predy):\n    predy[predy<0.5] = 0.0\n    predy[predy>=0.5] = 1.0\n    return (predy!=Y).mean()\n    \ndef multirestart(trainX,trainY,nhid,lam,ntries):\n    besterrsc = 1.0\n    for i in range(ntries):\n        Wts = trainneuralnet(trainX,trainY,nhid,lam)\n        errsc = errorrate(trainY,nneval(trainX,Wts))\n        if errsc<besterrsc:\n            returnWts = Wts\n            besterrsc = errsc\n    return returnWts\n    \nnhiddens = np.array([0,4,8,16])\nnhiddens = np.array([2,8,18,32])\nlams = np.logspace(-2.5,1.5,100)\nerates = np.empty([lams.shape[0],nhiddens.shape[0]])\nerates[:,:] = np.nan\n\n(f,ax) = setupfig()\n\n    \nfor ni, nhid in enumerate(nhiddens):\n    for li, lam in reversed(list(enumerate(lams))):\n        if nhid==0:\n            w = learnlogreg(trainX,trainY,lam)\n            predy = predictlogreg(validX,w)[:,np.newaxis]\n        else:\n            Wts = multirestart(trainX,trainY,nhid,lam,1) #trainneuralnet(trainX,trainY,nhid,lam)\n            predy = nneval(validX,Wts)\n        erates[li,ni] = errorrate(validY,predy)\n        \n        plotit(lams,nhiddens,erates,f,ax)")


# ## INTERPRET the Plot [5 points]
# <div style="color: #000000;background-color: #FFFFEE">
# How do you interpret the plot above?  How and why does the plot differ by number of hidden units?  By $\lambda$ value?  What parts of this plot agree with the material taught?  What parts do not?
# </div>

# <div style="color: #000000;background-color: #FFCCFF">
#     Grade: /5<br>
#     Comments:
#     </div>

# ### Your Answer Here
# 
# After meticulously going through my code, my a3 values or f values seem to all be the same. There is a slight difference in them about 16 decimal points after so 
# I tried to categorize them to the best of my ability. Because of this, I feel like my error rate is really high and I even messed around with lambda values and nothing
# really changes. I noticed on the plot, that the more hidden units we use, the lower the error is. I even tested the number of hidden units in the first problem, and I noticed 
# that the loss function starts off much bigger the more points there are as it goes down. As I increase my lamnda value, my f values get larger and larger. I increased my
# lambda to 1000 to test my function and it gave me a little more useable results versus when my lambda was 0.1 the results were only in the thousanths decimal place. I would
# say the only thing that this plot does not agree with the material taught is that all the f values are nondistuingashable and because of that, we are not getting accurate
# plots or y outputs. I think it does agree with the fact that regularization helps hone the weights in and gain a more clearer and accurate result. 
# 

# In[ ]:




