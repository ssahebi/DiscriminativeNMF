# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:22:00 2019
@author: mirza
"""

import numpy as np
import os
import csv
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans


def normal_s(path):
    with open(path + 'newmp.txt') as file:
        arr = [[float(digit) for digit in line.split(',')] for line in file]
    
    A = np.array(arr)
    
    maxA = np.amax(A)
    
    B = maxA - A
    
    lower = 0
    upper = 1.0
    
    m, n = B.shape
    
    print (m, n)
    
    l = np.ndarray.flatten(B)
    
    minl = float(np.amin(l))
    maxl = float(np.amax(l))
    
    l_norm = [ (upper - lower) * (x - minl) / (maxl - minl) + lower for x in l]
    
    nB = np.reshape(l_norm, (m, n))
    
    np.savetxt(path + "/nmp.csv", nB, delimiter=",")
    
    print (nB)


def normal_p(path):
    with open(path + 'p.txt') as file:
        arr = [[float(digit) for digit in line.split(',')] for line in file]
    
    A = np.array(arr)
    
    lower = 0
    upper = 1.0
    
    m, n = A.shape
    
    print (m, n)
    
    l = np.ndarray.flatten(A)
    
    minl = float(np.amin(l))
    maxl = float(np.amax(l))
    
    l_norm = [ (upper - lower) * (x - minl) / (maxl - minl) + lower for x in l]
    
    nA = np.reshape(l_norm, (m, n))
    
    np.savetxt(path + "/normal_p.csv", nA, delimiter=",")
    
    print (nA)

def create_matrix(path):
    
    p = []
    
    with open(path + "patternsTranslateFilterTFIDF.txt") as file:
        for line in file:
            p.append(line.strip())
        
    fw = open(path + 'newmp.txt', "w")
    
    for i in range(len(p)):
        row = ''
        for j in range(len(p)):
            row = row + ',' + str(nmlv(p[i].replace('_',''), p[j].replace('_','')))
            
        print (row)
        fw.write(row[1:] + '\n')
        
    fw.close()
        
def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def mlv(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                d = dis(seq1[x-1], seq2[y-1])
                matrix [x,y] = min(
                    matrix[x-1,y] + d,
                    matrix[x-1,y-1] + d,
                    matrix[x,y-1] + d
                )
    print (matrix)
    return (matrix[size_x - 1, size_y - 1])

def nmlv(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            d = dis(seq1[x-1], seq2[y-1])
            if seq1[x-1] == seq2[y-1]:
                if (x-2 > -1) and (y-2 > -1):
                    if seq1[x-2] == seq2[y-2]:
                        matrix [x,y] = min(
                    matrix[x-2, y-1] + d,
                    matrix[x-1, y] + d,
                    
                    matrix[x-1, y] + d,
                    matrix[x-1, y-2] + d,
                    
                    matrix[x-1, y-1],
                    matrix[x-2, y-2],
                    
                    matrix[x, y-1] + d,
                    matrix[x, y-2] + d
                    )
                        
                matrix [x,y] = min(
                    matrix[x-1, y] + d,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + d
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + d,
                    matrix[x-1,y-1] + d,
                    matrix[x,y-1] + d
                )
    print (matrix)
    return (matrix[size_x - 1, size_y - 1])


def dis(x, y):

    d = 1
    if (x.lower() == y.lower()
    or (x.islower() and y.islower())
    or (x.isupper() and y.isupper())
    ):
        d = 0.5
    
    return d

    
def column_center(X):
    avg = np.mean(X, axis=0)
    
    new_row = []
    
    for row in X:
        new_row.append(row - avg)
        
    return np.vstack(new_row)

def normal_similarity(path):
    
    with open(path + 'mp.txt') as file:
        array = [[float(digit) for digit in line.split('\t')] for line in file]
        
    p = np.array(array)
    
    n = column_center(p)
    
    np.savetxt(path + "/nmp.csv", n, delimiter=",")


def list_to_str(l):
    s = ''
    for i in l:
        s = s + ',' + str(i)
    return s[1:]

def list_to_str_tab(l):
    s = ''
    for i in l:
        s = s + '\t' + str(i)
    return s[1:]

def take_avg_vec(l):
    vecs = []
    for line in l:
        vec = [float(digit) for digit in line.split(',')]
        vecs.append(vec)
        
    return np.average(vecs, axis=0)

def take_std_vec(l):
    vecs = []
    for line in l:
        vec = [float(digit) for digit in line.split(',')]
        vecs.append(vec)
        
    return np.std(vecs, axis=0)

def print_label(k1, k2):
    s = ""
    for i in range(k1, k2+1):
        for j in range(1, i):
            s = s + "\t" + str(i) + "," + str(j)
    return s[1:]


def concateWcWd(path):
    
    with open(path + 'W1c.csv') as fw1c:
        w1c = list(csv.reader(fw1c, quoting=csv.QUOTE_NONNUMERIC))
    
    with open(path + 'W2c.csv') as fw2c:
        w2c = list(csv.reader(fw2c, quoting=csv.QUOTE_NONNUMERIC))
    
    with open(path + 'W1d.csv') as fw1d:
        w1d = list(csv.reader(fw1d, quoting=csv.QUOTE_NONNUMERIC))
        
    with open(path + 'W2d.csv') as fw2d:
        w2d = list(csv.reader(fw2d, quoting=csv.QUOTE_NONNUMERIC))
        
    w1cA = np.array(w1c)
    w2cA = np.array(w2c)
    w1dA = np.array(w1d)
    w2dA = np.array(w2d)
    
    w12cA = (w1cA + w2cA) /2
    
    w = np.concatenate((w12cA, w1dA, w2dA), axis = 1)
    
    np.savetxt(path + "W1cW2cW1dW2d.csv", w, delimiter=",")
    
    fw = open(path + "train-W.txt", "w")
    
    for row in w:
        out = ''
        for i in row:
            out = out + ',' + "{:.8f}".format(i)
        fw.write(out[1:] + '\n')
    
    fw1c.close()
    fw2c.close()
    fw1d.close()
    fw2d.close()
    fw.close()

def spectral(path, n):
    
    with open(path + 'W1cW2cW1dW2d.csv') as f:
        array2d = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))
        
    X = np.array(array2d)
    
    print (X.shape)
    
    ptrn = []
    with open(path + 'patternsTranslateFilterTFIDF.txt') as patterns:
        for p in patterns:
            t = p.split('\t')[0]
            ptrn.append(t)

    clustering = SpectralClustering(n_clusters=n, assign_labels="discretize", random_state=0).fit(X)
    
    pathc = path + "Spectral/"
    
    if (os.path.isdir(pathc) == False):
        os.mkdir(pathc)
        
    fw = open(pathc + "Spectral_" + str(n) + ".txt", "w")
    fwc = open(pathc + str(n) + ".txt", "w")
        
    for l in range(len(clustering.labels_)):
        fw.write(ptrn[l] + "\t" + str(clustering.labels_[l]) + "\n")
        fwc.write(str(clustering.labels_[l]) + "\n")

    fw.close()
    fwc.close()
    f.close()

def kmeans(path, n):
    
    with open(path + 'W1cW2cW1dW2d.csv') as f:
        array2d = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))
        
    X = np.array(array2d)
    
    print (X.shape)
    
    ptrn = []
    with open(path + 'patternsTranslateFilterTFIDF.txt') as patterns:
        for p in patterns:
            t = p.split('\t')[0]
            ptrn.append(t)

    clustering = KMeans(n_clusters=n, random_state=0).fit(X)
    
    pathc = path + "kmeans/"
    
    if (os.path.isdir(pathc) == False):
        os.mkdir(pathc)
        
    fw = open(pathc + "kmeans_" + str(n) + ".txt", "w")
        
    for l in range(len(clustering.labels_)):
        fw.write(ptrn[l] + "\t" + str(clustering.labels_[l]) + "\n")

    fw.close()
    f.close()
    
    

if __name__ == "__main__":
   
    path = ""
    
    # calculates the distance between each two patterns based on the modified levenshtein distance and build the distance matrix
    create_matrix(path)
    
    # normalizes the distance matrix and build similarity matrix
    normal_s(path)
    