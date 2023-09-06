import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
sym.init_printing(use_unicode=True)

def function (x):
    return(np.exp(-x)-x)


x = [0,0,1]
x[1] = ((x[2]-x[0])/2)


X = np.array(x)
#print(x)
Y = function(X)

def diff_matrix(nodos: np.array, valores: np.array):
    diff_matrix = np.zeros((len(nodos),len(valores)))
    diff_matrix[:,0] = valores

    for i in range(1,len(nodos)):
        for j in range(i,len(nodos)):
            diff_matrix[j,i] = diff_matrix[j,i-1]-diff_matrix[j-1,i-1]
    return diff_matrix

def Raiz(X,Y,itmax=100,precision=1e-8):

    error = 1.
    it = 0
    x0 = X[0]
    x1 = X[1]
    x2 = X[2]

    while error > precision and it < itmax:
        
        diff = diff_matrix(X,Y(np.array(X)))
        a = diff[len(X)-1,len(X)-1]
        b = diff[1,1] - (x0+x1)*a
        c = Y(x0) - x0*diff[1,1] + x0*x1*a

        if b >= 0:
            x3 = ((-2*c)/(b+((b**2) - 4*a*c)**(1/2)))
        else:
            x3 = ((-2*c)/(b-((b**2) - 4*a*c)**(1/2)))

        x0 = x3
        x1 = x0
        x2 = ((x1-x0)/2)

        X = [x0, x1, x2]


        if error > 1e-10:
            error = abs(Y(x3))
            
        it += 1
    
    if it == itmax:
        return False
    else:
        return x3
    
print(Raiz(X,function))
