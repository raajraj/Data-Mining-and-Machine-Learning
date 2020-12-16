#!/usr/bin/env python
# coding: utf-8

# # CS 171 PS 1
# # Due: Sunday, October 18, 2020 @ 11:59pm

# In[1]:


## Read *all* cells carefully and answer all parts (both text and missing code)


# ### Enter your information below:
# 
# <div style="color: #000000;background-color: #EEEEFF">
#     Your Name (submitter):  Raajitha Rajkumar<br>
#     Your student ID (submitter): 862015848
#     
# <b>By submitting this notebook, I assert that the work below is my own work, completed for this course.  Except where explicitly cited, none of the portions of this notebook are duplicated from anyone else's work or my own previous work.</b>
# </div>
# 

# ## Overview
# 
# This problem set deals with IMDB review data (from [here](https://ai.stanford.edu/~amaas/data/sentiment/)).  This dataset consists of reviews with either poor (<=4) or good (>=7) ratings.  The cells below load in training and testing data.  For each point, there are 1000 features, corresponding to the 1000 most common words in the reviews.  Each feature's value is from 0 to 6, with 0 indicating that the word did not appear, 1 indicating the word appeared once, 2 indicating the word appears between 2 and 4 times, and so on.  The corresponding y values are 0 for a poor rating and 1 for a good rating.
# 
# **We will treat these features as categorical** (That is, each feature's value is not treated as numeric, but as 7 different values that happen to be encoded using integers.)

# In[2]:


## THESE ARE THE ONLY LIBRARIES YOU MAY IMPORT!!
import numpy as np
import matplotlib.pyplot as plt

# below line just to make figures larger
plt.rcParams["figure.figsize"] = (20,10)


# In[3]:


def loaddata(fname):
    M = np.loadtxt(fname,dtype=float)
    np.random.shuffle(M)
    X = M[:,1:]
    threshs = [0,1,2,4,8,16,32,1024]
    for (i,(t1,t2)) in enumerate(zip(threshs[:-1],threshs[1:])):
        X[(X>t1) & (X<=t2)] = i
    Y = M[:,0]
    Y[Y<=0] = 0 # data is originally +1, -1
    return (X,Y)


# In[ ]:


(trainX,trainY) = loaddata('train.txt')
(testX,testY) = loaddata('test.txt')


# <div style="color: #000000;background-color: #FFFFEE">
#     <font size=+2>Question 1:</font> <font size=+1>(4 points)</font>
#     
# In the cell below, plot a grid of histograms (5 columns, 4 rows)
# Each histogram should be the distribution of a different feature (so you will be plotting just the first 20 features).  The histograms should have two sets of bars (each in their own color): ones for examples from class 0 and one for examples from class 1.  You want side-by-side histograms, each with 7 bars (for 14 bars in two colors). Be sure to give a title to each plot with the feature number.  Use only the training data for these histograms.
#     
# Hint: look up pyplot's [subplot](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplot.html) and [hist](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html) functions
#     
# As an example, the histogram in the upper left, corresponding to feature 0, should look like
#     
# ![feature0.png](attachment:5f4d1b5e-a2de-4cd4-979c-b7a00ea83fa1.png)  
# </div>

# In[1]:


### YOUR CODE HERE
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,10)


# In[2]:


def loaddata(fname):
    M = np.loadtxt(fname,dtype=int)
    np.random.shuffle(M)
    X = M[:,1:]
    threshs = [0,1,2,4,8,16,32,1024]
    for (i,(t1,t2)) in enumerate(zip(threshs[:-1],threshs[1:])):
        X[(X>t1) & (X<=t2)] = i
    Y = M[:,0]
    Y[Y<=0] = 0 # data is originally +1, -1
    return (X,Y)


# In[3]:


(trainX,trainY) = loaddata('train.txt')
(testX,testY) = loaddata('test.txt')


# In[9]:


data = np.vstack([trainX[:, :20].T, trainY[:]]).T
feature_good = data[:, :-1][data[:, -1]==0]
feature_bad = data[:, :-1][data[:, -1]==1]
histograms = plt.figure()
#for-loop to print 20 histograms
for i in range(20):
    feature = plt.subplot(4, 5, i+1)
    bins = [0,1,2,3,4,5,6]

    feature.hist([feature_good[:, i], feature_bad[:, i]], bins = bins, color = ['blue','orange'],label = ['bad', 'good'])
    feature.legend()
    feature.set_title('feature ' + str(i+1))


# <div style="color: #000000;background-color: #FFFFEE">
#     <font size=+2>Question 2:</font> <font size=+1>(4 points)</font>
#     
# For the 20 features above, based on the histograms you plotted, which would the most helpful three features for classifying this dataset using naive Bayes?  <b>WHY?</b>
# </div>
#     

# ### YOUR ANSWER HERE
# The most helpful 3 features for the dataset would probably be the histograms with the most normal 
# distribution such as feature 3, 4, and 5. Each feauture can actually be technically independent from
# the others. If each histogramrepresents one word that was used in reviews than it doesn't really 
# matter because each histogram represents different data. Although, the histograms with MORE data should
# be used since those words are appearing more in each review. We should probably use the data sets
# that appear more than less when classifying data. 
# 

# <div style="color: #000000;background-color: #FFFFEE">
#     <font size=+2>Question 3:</font> <font size=+1>(10 points)</font>
#         
# Complete the two functions in the two cells below.
# 
# The first trains/learns a naive Bayes classifier.  The second predicts the classes for a set of examples based on the model learned.
#     
# Hint:  Test your code on the example from the slides in class.  You'll need to generate the data matrix yourself and your own testing examples.
#     
# Hint 2: You will have to deal with counts that are 0.  Leaving them as zero will result in 0 probabilities that will cause problems.  A standard way to handle this is to add 1 to all counts.  For frequent feature values, it does not change things much.  For infrequent values, it keeps them away from 0 and admits that they might happen more often.
# </div>

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

def learnnb(X,Y):
    # X is shape (m,n) (m data points, each with n features).  It has integer values from 0 to maxfeatval (inclusive)
    # Y is shape (m,) (m data points).  It has values of either 0 or 1 (class "0" or class "1")
    
    
    # this function is to return a pair (priorp,condp)
    # where priorp is of shape (2,) and has the prior probability of each of the two classes
    # and condp is of shape (n,maxfeatval+1,2) and has the conditional probabilities for the naive Bayes classifier

   ## YOUR CODE HERE

    #priorp
    m = len(Y) #number of data points
    ones = 0
    zeroes = 0
    for i in range(m):
        if(Y[i]==1):
            ones = ones + 1
        else:
            zeroes = zeroes + 1
    priorpX = ones/m
    priorpY = zeroes/m
    priorp = (priorpX, priorpY)
    
    
    #condp
    maxfeatval = np.amax(X)
    n = len(X[0]) 
    condpa = np.empty(shape=(maxfeatval,2),dtype= float)
    condp = np.empty(shape=(n,maxfeatval+1,2),dtype = float)
    totalsum = 0.00
    counterz = 1.000;
    countero = 1.000;
    for i in range(n):
        for j in range(maxfeatval):
            for k in range(m):
                if(X[k][i] == j):
                    if(Y[k] == 0):
                        counterz = counterz + 1.00   
                    else:
                        countero = countero + 1.00
            counter = [counterz, countero]
            condpa[j] = counter
            counterz = 1.000;
            countero = 1.000;         
        for c in range(maxfeatval):
            
            totalsum = totalsum + condpa[c][0] + condpa[c][1]
        
        for s in range(maxfeatval):
            zero_attr = condpa[s][0]/totalsum
            one_attr = condpa[s][1]/totalsum
            combined_attr = zero_attr + one_attr
            reverse_zero = zero_attr/combined_attr
            reverse_one = one_attr/combined_attr 
            condp_zero = (combined_attr/priorpY)*reverse_zero
            condp_one = (combined_attr/priorpX)*reverse_one
            condp[i][s] = np.array([[[condp_zero, condp_one]]])           
    
    return (priorp,condp)  ## or whatever they are named in your code
    
X = [[1, 2, 3],[2, 3, 4],[5, 6, 7], [1, 2, 3],[1, 4, 5]]
Y = [0,0,1,1,0]
(prior, cond) = learnnb(X, Y)
for item in prior:
    print(item)
for item in cond:
    for it in item:
        print(it)
    print()
    


# In[5]:


def prednb(X,model):
    # X is of shape (m,n) (m data points, each with n features).
    # model is the pair (priorp,condp), as returned from learnnb
    # should return something of shape (m,) which is an array of 0s and 1s, indicating
    # the predicted (most probable under NB) class for each of the examples in X
    (priorp,condp) = model
    
    ## YOUR CODE HERE
    zero_pp = priorp[0]
    one_pp = priorp[1]
    prob_zero = 1;
    prob_one = 1;
    m = len(X)
    n = len(X[0])
    final = []
    for i in range(m):
        for j in range(n):
            prob_zero = prob_zero*condp[j][X[i][j]][0]
            prob_one = prob_one*condp[j][X[i][j]][1]
        prob_zero = prob_zero*zero_pp
        prob_one = prob_one*one_pp
        if(prob_zero > prob_one):
            final.append(0)
        else:
            final.append(1)
    return final

X = [[1, 2, 3],[2, 3, 4],[5, 6, 7], [1, 2, 3],[1, 4, 5]]
model = (prior, cond)
result = prednb(X, model)
for item in result:
    print(item)
        
        


# <div style="color: #000000;background-color: #FFFFEE">
# <font size=+2>Question 4:</font> <font size=+1>(2 points)</font>
# The code below trains a naive Bayes classifier and then tests it on the testing examples and reports the error rate.
#     
# Run the code.  [note, just because your code runs on this example, does not mean it is correct; for instance, both classes are equally likely a priori in this example, which is not always true.]
#     
# Then answer the question, "<b>Is this error rate good?  How do you know?</b>"
# </div>
# 

# In[9]:


def errorrate(predY,trueY):
    if len(predY)>1:
        predY = predY[:]
    if len(trueY.shape)>1:
        trueY = trueY[:,0]
    return (predY!=trueY).mean()

model = learnnb(trainX,trainY)
predY = prednb(testX,model)
print(errorrate(predY,testY))


# ### YOUR ANSWER HERE

# For the error rate being 0.5, I think it is fairly good considering that there are only two 
# classifiers. We have the probability of getting 1 and the probability of getting 0. There is a 
# fifty percent chance that it will be classified as 1 or 0 and therefore a fifty percent chance that
# it can be considered wrong. I think that even though the chances of getting a mistake are half and
# may be good or not good, the error rate is very accurate to the true chances of getting inaccurate
# results. 
