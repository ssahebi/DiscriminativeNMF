# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:52:44 2020

@author: mirzaei
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import os

def gd_structure(k, kc, pathin, pathout, alpha, beta, delta, ep, grid, epoc):
    
    #This method is the implementation of the final model with applying structure
    
    with open(pathin + 'lg/l.txt') as file:
        array2dX1 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X1 = np.array(array2dX1)
    
    with open(pathin + 'lg/h.txt') as file:
        array2dX2 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X2 = np.array(array2dX2)
    
    with open(pathin + 'nmp.csv') as file:
        arrayS = [[float(digit) for digit in line.split(',')] for line in file]
        
    S = np.array(arrayS)
    
    m1,n1 = X1.shape
    m2,n2 = X2.shape
    
    W1 = np.random.rand(m1,k)
    H1 = np.random.rand(n1,k)
    
    W2 = np.random.rand(m2,k)
    H2 = np.random.rand(n2,k)
    
    W1 = W1 / 10.0
    H1 = H1 / 10.0
    
    W2 = W2 / 10.0
    H2 = H2 / 10.0
    
    index = []
    errX1 = []
    errX2 = []
    errX1X2 = []
    errSqrC = []
    errD = []
        
    errS = []
    
    eps_list = []
    
    gama = 3.0 - (alpha + beta)
    
    reg = 0.01

    for e in range(epoc):
        learning_rate = 0.005/np.sqrt(e+1)
        learning_rate_c = 0.005/np.sqrt(e+1)
        learning_rate_h = 0.005/np.sqrt(e+1)
    
        W1c = W1[:,:kc]
        W1d = W1[:,kc:]
        
        H1c = H1[:,:kc]
        H1d = H1[:,kc:]
        
        W2c = W2[:,:kc]
        W2d = W2[:,kc:]
        
        H2c = H2[:,:kc]
        H2d = H2[:,kc:]
        
        W = np.concatenate((W1, W2), axis = 1)
        
        grad_w1c = (-2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1c)
        + 2 * alpha * (W1c - W2c)
        + 2 * reg * W1c
        - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W1c)
        ) 

        W1cn = W1c - learning_rate_c * grad_w1c
        
        grad_w2c = (-2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2c)
        - 2 * alpha * (W1c - W2c)
        + 2 * reg * W2c
        - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W2c)
        )
            
        W2cn = W2c - learning_rate_c * grad_w2c
        
        grad_w1d = (-2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1d) 
        + 2 * beta * np.dot(W2d, np.dot(W1d.T,W2d))
        + 2 * reg * W1d
        - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W1d)
        )
        
        W1dn = W1d - learning_rate * grad_w1d
        
        grad_w2d = (-2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2d) 
        + 2 * beta * np.dot(W1d, np.dot(W1d.T,W2d))
        + 2 * reg * W2d
        - 4 * delta * np.dot((S - eps * np.dot(W, W.T)), W2d)
        )
        
        W2dn = W2d - learning_rate * grad_w2d
        
        grad_h1 = (-2 * gama * np.dot(np.transpose(X1 - np.dot(W1, H1.T)), W1) 
        + 2 * reg * H1
        )
        
        H1n = H1 - learning_rate_h * grad_h1
        
        grad_h2 = (-2 * gama * np.dot(np.transpose(X2 - np.dot(W2, H2.T)), W2) 
        + 2 * reg * H2
        )
        
        H2n = H2 - learning_rate_h * grad_h2
        
        grad_eps = delta * np.sum(np.multiply( (-2 * np.dot(W, W.T)) , (S - eps * np.dot(W, W.T)) ))
        
        eps = eps - learning_rate * grad_eps
        
        eps_list.append(eps)
        
        W1n = np.concatenate((W1cn,W1dn), axis = 1)
        W2n = np.concatenate((W2cn,W2dn), axis = 1)
        
        W1n[W1n<0] = 0
        H1n[H1n<0] = 0

        W2n[W2n<0] = 0
        H2n[H2n<0] = 0

        errorX1 = error(X1, np.dot(W1, H1.T))
        
        errorX2 = error(X2, np.dot(W2, H2.T))
        
        errorSqrC = lossfuncSqr(W1cn, W2cn)
        
        errorD = lossfuncD(np.transpose(W1dn), W2dn)
        
        errorS = error(S, eps * np.dot(W, W.T))
        
        index.append(e)
        
        errX1.append(errorX1)

        errX2.append(errorX2)
        
        errX1X2.append((errorX1 + errorX2)/2)
        
        errSqrC.append(errorSqrC)
        
        errD.append(errorD)
        
        errS.append(errorS)
        
        W1 = W1n
        H1 = H1n
        
        W2 = W2n
        H2 = H2n
                
        if (e % 10 == 0):
            print (e)
            
        
    mode = 'write'
    #mode = 'test'
    
    if (mode == 'write'):        
        if (os.path.isdir(pathout) == False):
            os.mkdir(pathout)
        pathk = pathout + "k" + str(k)
        if (os.path.isdir(pathk) == False):
            os.mkdir(pathk)
        
        pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)
        if (os.path.isdir(pathkc) == False):
            os.mkdir(pathkc)
        
        if (grid == 'test'):
            np.savetxt(pathkc + "/W1.csv", W1, delimiter=",")
            np.savetxt(pathkc + "/W2.csv", W2, delimiter=",")
            
            np.savetxt(pathkc + "/W1c.csv", W1c, delimiter=",")
            np.savetxt(pathkc + "/W2c.csv", W2c, delimiter=",")
            np.savetxt(pathkc + "/W1d.csv", W1d, delimiter=",")
            np.savetxt(pathkc + "/W2d.csv", W2d, delimiter=",")
            
            np.savetxt(pathkc + "/H1.csv", H1, delimiter=",")
            np.savetxt(pathkc + "/H2.csv", H2, delimiter=",")
            
            np.savetxt(pathkc + "/H1c.csv", H1c, delimiter=",")
            np.savetxt(pathkc + "/H2c.csv", H2c, delimiter=",")
            np.savetxt(pathkc + "/H1d.csv", H1d, delimiter=",")
            np.savetxt(pathkc + "/H2d.csv", H2d, delimiter=",")        
        
        fw = open(pathkc + '/err.txt', "w")
        
        fwx1 = open(pathkc + '/errors-X1.txt', "w")
        fwx2 = open(pathkc + '/errors-X2.txt', "w")
        
        fwe = open(pathkc + '/eps.txt', "w")
            
        for i in errX1:
            fwx1.write(str(i) + '\n')
                
        for i in errX2:
            fwx2.write(str(i) + '\n')
            
        for i in eps_list:
            fwe.write(str(i) + '\n')
            
        errX1mae = errorMAE(X1, np.dot(W1, H1.T))
        errX2mae = errorMAE(X2, np.dot(W2, H2.T))
  
        
        fw.write("Error X1 RMSE: " + str(errX1[-1]) + " MAE: " + str(errX1mae) + "\n")
        fw.write("Error X2 RMSE: " + str(errX2[-1]) + " MAE: " + str(errX2mae) + "\n")
        fw.write("Error X1+X2: " + str(errX1X2[-1]) + "\n")
        fw.write("Error C: " + str(errSqrC[-1]) + "\n")
        fw.write("Error D: " + str(errD[-1]) + "\n")
        fw.write("Error S: " + str(errS[-1]) + "\n")
        fw.write("Eps: " + str(eps))
        
            
        fw.close()
        fwx1.close()
        fwx2.close()
        fwe.close()
        
        pathk = pathout + str(k)
        
        err1 = error(X1, np.dot(W1, H1.T))
        fw1 = open(pathk + 'err1.txt', "w")
        fw1.write(str(err1))
        fw1.close()
        
        err2 = error(X2, np.dot(W2, H2.T))
        fw2 = open(pathk + 'err2.txt', "w")
        fw2.write(str(err2))
        fw2.close()
        
        errs = error(S, np.dot(W,W.T))
        fw3 = open(pathk + 'errs.txt', "w")
        fw3.write(str(errs))
        fw3.close()
        

    pathk = pathout + "k" + str(k)
    pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)
    print (pathkc)
    
    
    #mode = 'test'
    mode = 'write'
    
    #sns.set()
    
    plt.figure()
    plt.plot(index,eps_list)
    plt.title('Error eps')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    if mode == 'write':
        plt.savefig(pathkc + "/Eps.png")
        plt.savefig(pathkc + "/Eps.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    
    plt.figure()
    plt.plot(index,errX1)
    plt.title('Error X1')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX1.png")
        plt.savefig(pathkc + "/ErrorX1.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errX2)
    plt.title('Error X2')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX2.png")
        plt.savefig(pathkc + "/ErrorX2.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errSqrC)
    plt.title('Error C')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorC.png")
        plt.savefig(pathkc + "/ErrorC.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errD)
    plt.title('Error D')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorD.png")
        plt.savefig(pathkc + "/ErrorD.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errS)
    plt.title('Error S')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorS.png")
        plt.savefig(pathkc + "/ErrorS.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
            
    print ('plot')
    plt.show()

def gd_eps_no_structure(k, kc, pathin, pathout, alpha, beta, grid, epoc):
    
    with open(pathin + 'l.txt') as file:
        array2dX1 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X1 = np.array(array2dX1)
    
    with open(pathin + 'h.txt') as file:
        array2dX2 = [[float(digit) for digit in line.split(',')] for line in file]
    
    X2 = np.array(array2dX2)
    
    m1,n1 = X1.shape
    m2,n2 = X2.shape
    
    X1_size = m1 * n1
    X2_size = m2 * n2
    
    W1 = np.random.rand(m1,k)
    H1 = np.random.rand(n1,k)
    
    W2 = np.random.rand(m2,k)
    H2 = np.random.rand(n2,k)
    
    W1 = W1 / 10.0
    H1 = H1 / 10.0
    
    W2 = W2 / 10.0
    H2 = H2 / 10.0
    
    index = []
    errX1 = []
    errX2 = []
    errX1X2 = []
    errSqrC = []
    errD = []
    
    #C
    #alpha
    #D
    #beta
    
    #X1X2
    gama = 2.0 - (alpha + beta)
    #gama
    
    reg = 0.01

    for e in range(epoc):
        learning_rate = 0.01/np.sqrt(e+1)
    
        W1c = W1[:,:kc]
        W1d = W1[:,kc:]
        
        H1c = H1[:,:kc]
        H1d = H1[:,kc:]
        
        W2c = W2[:,:kc]
        W2d = W2[:,kc:]
        
        H2c = H2[:,:kc]
        H2d = H2[:,kc:]
        
        grad_w1c = (2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1c)
            + 2 * alpha * (W1c - W2c)
            + 2 * reg * W1c
            )
        
        
        W1cn = W1c - learning_rate * grad_w1c
        
        grad_w2c = (2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2c)
            - 2 * alpha * (W1c - W2c)
            + 2 * reg * W2c
            )
        
        W2cn = W2c - learning_rate * grad_w2c
        
        grad_w1d = (2 * gama * np.dot((np.dot(W1, H1.T) - X1), H1d) 
            + 2 * beta * np.dot(W2d, np.dot(W1d.T,W2d))
            + 2 * reg * W1d
            )
        
        W1dn = W1d - learning_rate * grad_w1d
        
        grad_w2d = (2 * gama * np.dot((np.dot(W2, H2.T) - X2), H2d) 
            + 2 * beta * np.dot(W1d, np.dot(W1d.T,W2d))
            + 2 * reg * W2d
            )
        
        W2dn = W2d - learning_rate * grad_w2d
        
        grad_h1 = -2 * gama * np.dot(np.transpose(X1 - np.dot(W1, H1.T)), W1) + 2 * reg * H1
        H1n = H1 - learning_rate * grad_h1
        
        grad_h2 = -2 * gama * np.dot(np.transpose(X2 - np.dot(W2, H2.T)), W2) + 2 * reg * H2
        H2n = H2 - learning_rate * grad_h2
        
        W1n = np.concatenate((W1cn,W1dn), axis = 1)
        W2n = np.concatenate((W2cn,W2dn), axis = 1)
        
        W1n[W1n<0] = 0
        H1n[H1n<0] = 0

        W2n[W2n<0] = 0
        H2n[H2n<0] = 0

        errorX1 = error(X1, np.dot(W1, H1.T))
        errorX2 = error(X2, np.dot(W2, H2.T))
        errorSqrC = lossfuncSqr(W1cn, W2cn)
        errorD = lossfuncD(np.transpose(W1dn), W2dn)
                
        index.append(e)
        
        errX1.append(errorX1)
        errX2.append(errorX2)
        errX1X2.append((errorX1/X1_size + errorX2/X2_size )/2)
        errSqrC.append(errorSqrC)
        errD.append(errorD)
        
        W1 = W1n
        H1 = H1n
        
        W2 = W2n
        H2 = H2n
        
        if (e % 10 == 0):
            print (e)
            
    #mode = 'test'
    mode = 'write'
    
    if (mode == 'write'):        
        if (os.path.isdir(pathout) == False):
            os.mkdir(pathout)
        pathk = pathout + "k" + str(k)
        if (os.path.isdir(pathk) == False):
            os.mkdir(pathk)
        
        pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)    
        if (os.path.isdir(pathkc) == False):
            os.mkdir(pathkc)
        
        #print 'Mode write'
        if (grid == 'test'):
            
            fwp = open(pathkc + '/params.txt', "w")
            fwp.write('K: ' + str(k) + '\nKc: ' + str(kc) + '\nalpha: ' + str(alpha) + '\nbeta: ' + str(beta) + '\ngama: ' + str(gama) + '\nreg: ' + str(reg))
            fwp.close()
        
            np.savetxt(pathkc + "/W1.csv", W1, delimiter=",")
            np.savetxt(pathkc + "/W2.csv", W2, delimiter=",")
            
            np.savetxt(pathkc + "/W1c.csv", W1c, delimiter=",")
            np.savetxt(pathkc + "/W2c.csv", W2c, delimiter=",")
            np.savetxt(pathkc + "/W1d.csv", W1d, delimiter=",")
            np.savetxt(pathkc + "/W2d.csv", W2d, delimiter=",")
            
            np.savetxt(pathkc + "/H1.csv", H1, delimiter=",")
            np.savetxt(pathkc + "/H2.csv", H2, delimiter=",")
            
            np.savetxt(pathkc + "/H1c.csv", H1c, delimiter=",")
            np.savetxt(pathkc + "/H2c.csv", H2c, delimiter=",")
            np.savetxt(pathkc + "/H1d.csv", H1d, delimiter=",")
            np.savetxt(pathkc + "/H2d.csv", H2d, delimiter=",")
            
        
        fw = open(pathkc + '/err.txt', "w")
        
        fwx1 = open(pathkc + '/errors-X1.txt', "w")
        fwx2 = open(pathkc + '/errors-X2.txt', "w")
        
        for i in errX1:
            fwx1.write(str(i) + '\n')
                
        for i in errX2:
            fwx2.write(str(i) + '\n')
        
        errX1mae = errorMAE(X1, np.dot(W1, H1.T))
        errX2mae = errorMAE(X2, np.dot(W2, H2.T))
        
        fw.write("Error X1 RMSE: " + str(errX1[-1]) + " MAE: " + str(errX1mae) + "\n")
        fw.write("Error X2 RMSE: " + str(errX2[-1]) + " MAE: " + str(errX2mae) + "\n")
        fw.write("Error X1+X2: " + str(errX1X2[-1]) + "\n")
        fw.write("Error C: " + str(errSqrC[-1]) + "\n")
        fw.write("Error D: " + str(errD[-1]) + "\n")
        
        errorX1 = np.abs(X1 - np.dot(W1,np.transpose(H1)))
        errorX2 = np.abs(X2 - np.dot(W2,np.transpose(H2)))
            
        fw.close()
        fwx1.close()
        fwx2.close()
        
        pathk = pathout + str(k)
        
        err1 = error(X1, np.dot(W1, H1.T))
        fw1 = open(pathk + 'err1.txt', "w")
        fw1.write(str(err1))
        fw1.close()
        
        err2 = error(X2, np.dot(W2, H2.T))
        fw2 = open(pathk + 'err2.txt', "w")
        fw2.write(str(err2))
        fw2.close()


    pathk = pathout + "k" + str(k)
    pathkc = pathk + "/c" + str(kc) + "d" + str(k-kc)
    print (pathkc)
    
    
    #mode = 'test'
    mode = 'write'
    
    plt.figure()
    plt.plot(index,errX1)
    plt.title('Error X1')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX1.png")
        plt.savefig(pathkc + "/ErrorX1.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errX2)
    plt.title('Error X2')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorX2.png")
        plt.savefig(pathkc + "/ErrorX2.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errSqrC)
    plt.title('Error C')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorC.png")
        plt.savefig(pathkc + "/ErrorC.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    plt.figure()
    plt.plot(index,errD)
    plt.title('Error D')
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    if mode == 'write':
        plt.savefig(pathkc + "/ErrorD.png")
        plt.savefig(pathkc + "/ErrorD.pdf")
        if (grid == 'grid'):
            plt.clf()
            plt.close()
    
    
    print ('plot')
    
    plt.show()
    plt.rcParams.update({'font.size': 10})

def error(A, B):
    return np.sqrt(np.mean((A - B) ** 2))

def errorMAE(A, B):
    return np.mean(np.abs(A - B))

def lossfuncSqr(X,Xn):
    m,n = X.shape
    sum = 0
    e = 0.0
    for i in range(m):
        for j in range(n):
            sum += math.pow(X[i,j] - Xn[i,j], 2)
            e += 1
    return math.sqrt(sum / e)

def lossfuncD(X,Xn):
    sum = 0
    e = 0.0
    Y = np.dot(X,Xn)
    m,n = Y.shape
    for i in range(m):
        for j in range(n):
            sum += abs(Y[i,j])
            e += 1
    return sum/e


def grid_search_no_structure(pathin, pathout):
    
    a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    kc = 0
    for alpha in a:
        for beta in b:
                pathn = pathout + 'a' + str(alpha) + 'b' + str(beta) + '/'
                
                if (os.path.isdir(pathout) == False):
                    os.mkdir(pathout)
                if (os.path.isdir(pathn) == False):
                    os.mkdir(pathn)                
                for k in range(10, 21, 1):
                    for kc in range (1, k):                        
                        try:
                            gd_eps_no_structure(k, kc, pathin, pathn, alpha, beta, grid = 'grid', epoc = 100)
                        except Exception as e:
                            fwerr = open(pathout + '/errors' + '_a' + str(alpha) + '_b' + str(beta) +  '.txt', "w")
                            fwerr.write('alpha: ' + str(alpha) + '\tbeta: ' + str(beta) + '\tk: ' + str(k) + '\tkc: ' + str(kc)  + '\n' + str(e))
                            fwerr.close()                            
                    print ('K %s and Kc %s compeleted' %(k, kc))



def find_best_param_4_no_s(path):
    
    x1_error = []
    x2_error = []
    x12_error = []
    c_error = []
    d_error = []
    sum_all_error = []
    index = []
    real_index = []
    ix = 0
    
    x1_dict = {}
    x2_dict = {}
    x12_dict = {}
    c_dict = {}
    d_dict = {}
    sum_all_dict = {}
    
    x1_param = ''
    x2_param = ''
    x12_param = ''
    c_param = ''
    d_param = ''
    sum_all_param = ''
    
    x1_min = 1.0
    x2_min = 1.0
    x12_min = 1.0
    c_min = 1.0
    d_min = 1.0
    sum_all_min = 1.0
    
    dirs = os.listdir(path)
    
    for d in dirs:
        if os.path.isdir(path + d) and len(d) == 8:
            pathd = path + d + '/'
            print (d)
            dird = os.listdir(pathd)
            for k in dird:
                if os.path.isdir(pathd + k) and get_digits(k) < 22:
                    pathk = pathd + k + '/'
                    dirsk = os.listdir(pathk)
                    print pathk
                    for c in dirsk:
                        pathc = pathk + c + '/'
                        if os.path.isdir(pathc):
                            pathf = pathc + 'err.txt'
                            #print (pathf)
                            f = open(pathf, "r")
                            lines = f.readlines()
                                    
                            e1 = float(lines[0].split()[3])
                            e2 = float(lines[1].split()[3])
                            e12 = float(lines[2].split()[2])
                            ec = float(lines[3].split()[2])
                            ed = float(lines[4].split()[2])
                            sum_all = e1 + e2 + e12 + ec + ed
                            
                            x1_error.append(e1)
                            x2_error.append(e2)
                            x12_error.append(e12)
                            c_error.append(ec)
                            d_error.append(ed)
                            sum_all_error.append(sum_all)
                            
                            
                            comb = d + ' ' + k + ' ' + c
                            
                            x1_dict[e1] = comb
                            x2_dict[e2] = comb
                            x12_dict[e12] = comb
                            c_dict[ec] = comb
                            d_dict[ed] = comb
                            sum_all_dict[sum_all] = comb
                            
                            
                            if e1 < x1_min:
                                x1_min = e1
                                x1_param = comb
                            
                            if (e2 < x2_min):
                                x2_min = e2
                                x2_param = comb
                            
                            if (e12 < x12_min):
                                x12_min = e12
                                x12_param = comb
                            
                            if (ec < c_min):
                                c_min = ec
                                c_param = comb
                            
                            if (ed < d_min):
                                d_min = ed
                                d_param = comb
                            
                            
                            if (sum_all < sum_all_min):
                                sum_all_min = sum_all
                                sum_all_param = comb
                            
                            
                            ix = ix + 1
                            index.append(ix)
                            real_index.append('K='+ k.replace('k','') + ', Kc=' + c.split('d')[0].replace('c','') + ', Kd=' + c.split('d')[1].replace('d',''))
                                
                                    
            
    print (x1_min, x1_param)
    print (x2_min, x2_param)
    print (x12_min, x12_param)
    print (c_min, c_param)
    print (d_min, d_param)
    print (sum_all_min, sum_all_param)
    
    fw = open(path + '/min_error.txt', "w")
    fw.write(x1_param + ': ' + str(x1_min) + '\n')
    fw.write(x2_param + ': ' + str(x2_min) + '\n')
    fw.write(x12_param + ': ' + str(x12_min) + '\n')
    fw.write(c_param + ': ' + str(c_min) + '\n')
    fw.write(d_param + ': ' + str(d_min) + '\n')
    fw.write(sum_all_param + ': ' + str(sum_all_min))
    
    fw.close()
    
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams.update({'font.size': 24})
    
    plt.figure()
    plt.plot(index,x1_error)
    plt.title('Error X1')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorX1.png")
    
    plt.figure()
    plt.plot(index,x2_error)
    plt.title('Error X2')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorX2.png")
    
    plt.figure()
    plt.plot(index,x12_error)
    plt.title('Error X12')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorX12.png")
    
    plt.figure()
    plt.plot(index,c_error)
    plt.title('Error C')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorC.png")
    
    plt.figure()
    plt.plot(index,d_error)
    plt.title('Error D')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorD.png")
    
    plt.figure()
    plt.plot(index,sum_all_error)
    plt.title('Error Sum All')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(path + "/ErrorAll.png")
    
    plt.show()
    
    x1_error.sort()
    x2_error.sort()
    x12_error.sort()
    c_error.sort()
    d_error.sort()
    sum_all_error.sort()
    
    fwt = open (path + '/top10.txt', 'w')
    
    out1 = ''
    out2 = ''
    out12 = ''
    outc = ''
    outd = ''
    outa = ''
    
    for i in range(10):
        out1 = out1 + '\t' + x1_dict.get(x1_error[i]) + ': ' + str(x1_error[i])
        out2 = out2 + '\t' + x2_dict.get(x2_error[i]) + ': ' + str(x2_error[i])
        out12 = out12 + '\t' + x12_dict.get(x12_error[i]) + ': ' + str(x12_error[i])
        outc = outc + '\t' + c_dict.get(c_error[i]) + ': ' + str(c_error[i])
        outd = outd + '\t' + d_dict.get(d_error[i]) + ': ' + str(d_error[i])
        outa = outa + '\t' + sum_all_dict.get(sum_all_error[i]) + ': ' + str(sum_all_error[i])
        
    fwt.write('X1\t' + out1[1:] + '\nX2\t' + out2[1:] + '\nX12\t' + out12[1:] + '\nC\t' + outc[1:] + '\nD\t' + outd[1:] + '\nSum\t' + outa[1:])
    
    fwt.close()
                
def get_digits(s):
    c = ""
    for i in s:
        if i.isdigit():
            c += i
    return int(c)



if __name__ == "__main__":
    
    pathin = 'files/'
    pathout = pathin + 'DNMF/'
    
    k = 20
    kc = 10
    epoc = 1000
    alpha = 0.6
    beta = 0.1
    delta = 0.9
    
    gd_eps_no_structure(k, kc, pathin, pathout, alpha, beta, epoc)
    
    #grid_search_no_structure(pathin, pathin + 'grid/')
    
    #find_best_param_4_no_s(pathin + 'grid/')
    
