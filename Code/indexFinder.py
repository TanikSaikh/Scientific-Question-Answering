import os
import numpy as np 
import nltk
import re
import pandas as pd
import string

f = open("train.span", "w+")

df = pd.read_csv("Context-train.csv",sep = ',')
s  = df.iloc[0:,1:2]

dfA = pd.read_csv("Answer-train.csv",sep = ',')
ans = dfA.iloc[0:,1:2]

X = []
for j in s['Context']:
    X.append(j)
c = 0
A = []
for line in ans['Answer']:
    line = line.split()
    #c = c+1
    #print(type(line))
    #print("***********",c)
    A.append(line)



def find_sub_list(subl, l):
    #print(subl,len(subl))
    #print(l,len(l))
    ind_subl = [i for i in range(len(l)) if l[i] in subl]
    if(len(subl) == 1):
        #print("in if>>>>>>")
        #print("**********", ind_subl, ind_subl)
        return [str(ind_subl[0]), str(ind_subl[0])]
    else:
        #print("in else>>>>>>")
        return [str(ind_subl[0]), str(ind_subl[-1])]

for (i, j) in zip(X,A):
    print(i.split())
    print(j)
    print(find_sub_list(j,i.split()))
    #print(type(find_sub_list(j,i.split())))
    f.write(' '.join(find_sub_list(j,i.split())))
    f.write("\n")
    #for k in j:
        #print(k[0])
    c = c+1
    print("index is >>", c)
print(c)
#    for j in i:
 #       print(j)
f.close()
