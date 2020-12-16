#!/usr/bin/env python
# coding: utf-8

# # CS 171 / EE 142 Problem Set 4
# # Due Sunday, December 6, 2020 @ 11:59pm 

# ## Read *all* cells carefully and answer all parts (both text and code)

# ### Enter your information below:
# 
# <div style="color: #000000;background-color: #FFCCCC">
#     Your Name (submitter): Raajitha Rajkumar <br>
# Your student ID (submitter): 862015848
# </div>

# <div style="color: #000000;background-color: #FFCCFF">
#     Total Grade: /30<br>
#     Late Days on this assigment: 0<br>
#     Total Late Days so far: 5<br>
#     </div>

# <div style="color: #000000;background-color: #FFEEFF">
#     <font size=+2>Part I: Association Rules</font>
# </div>

# <div style="color: #000000;background-color: #FFFFEE">
#     <font size=+2>Question 1:</font> <font size=+1>(4 points)</font>
# 
# If there are $n$ items (or features), there are $3^n-2^{n+1}+1$ different possible association rules.  Prove this.
# 
# You need a _clear_ explanation.  Note, associate rules must have a non-empty set on the left-hand and right-hand sides.
# </div>

# ### Your Answer Here
# 
# If we have $n$ items, we automatically get $2^{n}$ subsets. Now it gets tricky because we have to combine all these subsets 
# in order to get all possible rules except we cannot just blatantly combine each set given. We have to generally find the sum from $i$ = 1 
# to $n$-1 of the combination of $n$ and $i$ (this is for the left side of the rules) multiplied by the sum from $j$ = 1 to $n - i$ 
# of the combination of $n$ - $i$ and $j$ (these are the right side of the rules).
# In more simpler terms.
# \begin{align*} \\
# R = \sum_{i=1}^{n-1} \left(\binom{n}{i}\sum_{j=1}^{n-i}\binom{n-i}{j}\right) \\ 
#  = \sum_{i=1}^{n-1} \binom{n}{i} 2^{n-i} - 1 \\
#  = \sum_{i=1}^{n-1} \binom{n}{i} 2^{n-i} - \sum_{i=1}^{n-1} \binom{n}{i} \\
# because \sum_{i=1}^{n-1} = 2^{n} - 1 \\
# R = \sum_{i=1}^{n-1} \binom{n}{i} 2^{n-i} - (2^{n} - 1) \\
# \end{align*} 
# 
# Let's take this equation and compare it to the series where...
# \begin{align*} \\
# \sum_{i=1}^{n-1} \binom{n}{i} x^{n-k} - x^{n} = (1+x)^{n} \\
# \end{align*}
# 
# It looks sort of familiar as in the previous derivation we are doing that with x = 2 
# i to n. So when we use it in the formula we get that part to equal to $3^{n}$
# \begin{align*} \\
# \sum_{i=1}^{n-1} \binom{n}{i} 2^{n-k} - 2^{n} = 3^{n} \\
# \end{align*}
# 
# Now that we have $3^{n}$ we can plug that in to replace $\sum_{i=1}^{n-1} \binom{n}{i} 2^{n-i}$.
# We also have to make sure to subtract $2^{n}$ since we added it in the first place
# to replace the expression.
# \begin{align*} \\
# R = 3^{n} - 2^{n} - 2^{n} + 1 \\
# R = 3^{n} - 2(2^{n}) + 1 \\
# R = 3^{n} - 2^{n+1} + 1 \\
# \end{align*}

# <div style="color: #000000;background-color: #FFCCFF">
#     Q1:<br>
#     Grade: /4<br>
#     </div>

# <div style="color: #000000;background-color: #FFFFEE">
#     <font size=+2>Question 2:</font> <font size=+1>(12 points)</font>
#     
# In this question, you will write code to do association rule learning, as described in class.
# 
# The items will be represented by numbers (for ease and speed) with a separate
# list of the names for each item.  `loaddata` (below) loads in a dataset and returns these three things: a list of the names of each item, a list of the examples, and the total number of items.  Each example is a set of numbers representing.  For example, for the toy problem in lecture, loaddata returns
# 
# `['Jurassic Park', 'Star Wars', 'Forrest Gump', 'Home Alone', 'Toy Story']`
# 
# `[[1, 2, 4], [1, 4], [1, 3, 4], [0, 1], [0, 3], [1, 3, 4], [0, 2, 3], [3], [1, 3, 4], [1]]`
# 
# `5`
# 
# You should use `set`s and `frozenset`s (core python data structures) in your code.  You can read more about them at https://docs.python.org/3/library/stdtypes.html#set
# 
# Write the functions `learnrules` and `writerules`, plus any additional helper functions you need.  Use the apriori algorithm to generate "large item lists" and the algorithm from class to find rules that meet the minimum support and confidence given.
# </div>

# In[67]:


import itertools
from itertools import combinations, chain #do not import anything else 
# (you may or may not use combinations -- up to you)

# prints out a set, nicely
# names is an optional list of the names for each of the (integer) items
def settostr(s,names=None):
    if names is None:
        elems = [str(e) for e in s]
    else:
        elems = [names[e] for e in s]
    return "{" + (", ".join(elems)) + "}"


# In[68]:


# loads in data from filename, assuming the file format used for this assignment
def loaddata(filename):
    with open(filename) as f:
        nitems = int(f.readline())
        names = [f.readline().strip() for i in range(nitems)]
        nrows = int(f.readline())
        data = [[int(s) for s in f.readline().split()] for i in range(nrows)]
        f.close()
        return (names,data,nitems)        


# In[69]:


# this function uses itertoold combinations to generate combinational sets for making the subsets
def generate_possible_combinations(R, size):
    subsets = list(map(set, itertools.combinations(R,size)))
    return subsets

# this function uses itertoold combinations to generate permutational sets for making the subsets
def generate_possible_permutations(R, size):
    subsets = list(map(set, itertools.permutations(R,size)))
    return subsets

# this function uses itertoold combinations to generate permutational sets for making the subsets
def generate_product(R, size):
    subsets = list(map(list, itertools.product(R,repeat = size)))
    return subsets

# most of this function is the beginning of Apriori-Gen as it generates
# this first individual layer of nodes before branching out
# this function as a whole generates all subsets shown from the lattice in the slides
def generate_subsets(numitems, d, smin):
    R = set()
    count = 0
    d_len = len(d)
    for i in range(numitems): # this iterates through the number of items
        for j in range(d_len): # this iterates through the list of sets of data
            ds = set(d[j])
            if(i in ds): # if the item is in the set of data, increase the count
                count = count + 1
        if((count/d_len) >= smin): # if the support is greater than the minimum, add it to the total set
            R.add(i) # add to R
        count = 0
    R = generate_lattice(d, smin, R, R, 1)
    return R

# this makes up the rest of Apriori_Gen to find the rest of the important or "black" nodes
# this function uses recursion to generate all the children for the final subsets
def generate_lattice(d, smin, R, children, size):
    print("generating subsets...takes more than 20 minutes")
    rlen = len(children)
    count = 0
    dlen = len(d)
    countgoal = smin*dlen
    curr_children = set()   
    size = size + 1
    
    # if the amount of children are 2 or greater, explore the children, else, lattice is done
    if(rlen >= 2):
        # if the children are frozensets, make them able to be used to generate combinations
        Rl = list(children)
        if(type(Rl[0]) is frozenset):
            children = make_usable(children)
        # generate possible combinations, in the slides, it should be the set of sets in that level
        psubsets = generate_possible_combinations(children, size)
        psubsetlen = len(psubsets)
        # iterate through the data and see which subset can be added to R
        for i in range(psubsetlen):
            for k in range(dlen):
                ds = set(d[k])
                if(psubsets[i].issubset(ds)):
                    count = count + 1         
                if(count >= countgoal):
                    stemp = frozenset(psubsets[i])
                    R.add(stemp) # add to R
                    curr_children.add(stemp)
                    k = dlen
            count = 0
            k = 0
        R = generate_lattice(d, smin, R, curr_children, size)
        return R
    return R

# this function takes a set of frozen sets and makes it into a set with all its elements
def make_usable(c):
    c = list(c)
    chl = list()
    for i in c:
        for j in i:
            chl.append(j)
    ch = set(chl)
    return ch

# this function takes a list or set and makes it into a list of individual sets
def makelist(S):
    S = list(S)
    for i in range(len(S)):
        if(type(S[i]) is not frozenset):
            S[i] = frozenset([S[i]]) 
    return S

def learnrules(numitems,data,minsupport,minconfidence):  
    count = 0
    xcount = 0
    rulepackage = list()
    dlen = len(data)
    
    # use Apriori-Gen to generate candidates, and turn it into a list to make rules
    S = generate_subsets(numitems, data, minsupport) 
    S = makelist(S)
    # iterate through all the data with possible rule combos and calculate confidence and support
    I = generate_product(S, 2)
    for i in I:
        if(len(i) == 2):
            if(i[0].isdisjoint(i[1])):
                i = list(i)
                 # iterates through data, i and j are candidates
                for k in data:
                    ds = set(k)
                    # calculate total x
                    if(i[0].issubset(ds)):
                        xcount = xcount + 1
                    # calculate total of x an y
                    if((i[0].issubset(ds)) & (i[1].issubset(ds))):
                        count = count + 1           
                if(xcount == 0):
                    xcount = 0   
                else:
                    conf = count/xcount # confidence
                    supp = count/dlen # support
                    if((conf >= minconfidence) & (supp >= minsupport)):
                        print("generating rules... takes more than 20 minutes")
                        rulepair = (i[0],i[1]) # make pair
                        rulepackage.append((supp, conf, rulepair)) # add to final rules
                count = 0
                xcount = 0
    return rulepackage


# In[70]:


def writerule(rule,itemnames):
    return settostr(rule[0],itemnames) + " => " + settostr(rule[1],itemnames)

# helper function to sort confidence
def sortSecond(val):
    return val[1]

def writerules(rules,data,itemnames):
    rules.sort(key = sortSecond, reverse = True)
    print("\n")
    for x in rules:
        (su, co, ru) = x
        print('%7.4f'%(su) + " " + '%7.4f'%(co) + "    " + writerule(ru, itemnames))
        
    return 


# In[71]:


# prints the rule set
def printruleset(datasetfilename,minsupport,minconfidence):
    (itemnames,data,numitems) = loaddata(datasetfilename)
    rules = learnrules(numitems,data,minsupport,minconfidence)
    writerules(rules,data,itemnames)


# In[72]:


## toy dataset example
printruleset('toymovies.txt',0.3,0.5)
#['Jurassic Park', 'Star Wars', 'Forrest Gump', 'Home Alone', 'Toy Story']
#''' output should look like
#0.5000  1.0000    {Toy Story} => {Star Wars}             4 -> 1
#0.3000  1.0000    {Star Wars, Home Alone} => {Toy Story} 13 -> 4
#0.3000  1.0000    {Home Alone, Toy Story} => {Star Wars} 34 -> 1
#0.5000  0.7143    {Star Wars} => {Toy Story}             1 -> 4
#0.3000  0.6000    {Star Wars, Toy Story} => {Home Alone} 14 -> 3
#0.3000  0.6000    {Toy Story} => {Home Alone}            4 -> 3
#0.3000  0.6000    {Toy Story} => {Star Wars, Home Alone} 4 -> 13
#0.3000  0.5000    {Home Alone} => {Toy Story}            3 -> 4
#0.3000  0.5000    {Home Alone} => {Star Wars, Toy Story} 3 -> 14
#0.3000  0.5000    {Home Alone} => {Star Wars}            3 -> 1
#'''


# In[73]:


get_ipython().run_cell_magic('time', '', "# the full groceries answer (should take under a minute to run)\nprintruleset('groceries.txt',0.01,0.5)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# a different groceries dataset, looking for rules with less support, but higher confidence (should take under a minute to run)\nprintruleset('groceries2.txt',0.001,0.7)")


# <div style="color: #000000;background-color: #FFCCFF">
#     Q2:<br>
#     Grade: /12<br>
#     </div>

# <div style="color: #000000;background-color: #FFEEFF">
#     <font size=+2>Part II: Clustering</font>
# </div>

# <div style="color: #000000;background-color: #FFFFEE">
#     <font size=+2>Question 3:</font> <font size=+1>(4 points)</font>
# 
# The code below plots 6 points (in 2D feature space) and the associated dendrograms for
# three types of linkage definitions: single, average, and complete.
# 
# However, for these six points, all three dendrograms are almost the same.  While the levels at which points are merged differ, the clusters generated are the same.
# 
# Change the points below (`pts`) so that each of the three linkages produces a different heirarchical clustering.
# </div>

# In[ ]:


pts = [[0,0],[0,2.5],[1.75,1.75],[2,4],[3.5,0.5],[5,2]] ## Change only this line (but keep 6 points)
pnames = ['A','B','C','D','E','F']


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.cluster import hierarchy
from scipy.spatial import distance
import matplotlib.pyplot as plt

ctypes = ['single','average','complete']

plt, axes = plt.subplots(1,len(ctypes)+1,figsize=(4+4*len(ctypes),4))

axes[0].scatter([x[0] for x in pts],[x[1] for x in pts])
for i,name in enumerate(pnames):
    axes[0].annotate(name,(pts[i][0],pts[i][1]))
axes[0].axis('equal')
axes[0].set_title('points')
    
for i,ctype in enumerate(ctypes):
    Z = hierarchy.linkage(distance.pdist(pts),ctype)
    hh = hierarchy.dendrogram(Z,ax=axes[i+1],labels=pnames)
    axes[i+1].set_title(ctype+ " linkage")


# <div style="color: #000000;background-color: #FFCCFF">
#     Q3:<br>
#     Grade: /4<br>
#     </div>

# In[ ]:




