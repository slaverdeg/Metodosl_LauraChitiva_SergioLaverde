import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
sym.init_printing(use_unicode=True)
"""
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
    
print(Raiz(X,function))"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym



x = np.linspace(-1,1,50)

def Function(x): 
    return np.exp(-x)-x

def Derivative(f,x,h=1e-6):
    return (f(x+h)-f(x-h))/(2*h)


def Newton(f,df,xn,itmax=100,precision=1e-8):
    error = 1.
    it = 0
    
    while error > precision and it < itmax:
        
        try:
            
            xn1 = xn - f(xn)/df(f,xn)
            error = np.abs(f(xn)/df(f,xn))
            
        except ZeroDivisionError:
            print('Division por cero')
            
        xn = xn1
        it += 1
    
    if it == itmax:
        return False
    else:
        return xn
    
def Raices(x, tolerancia=10): 
    Roots = np.array([])
    
    for i in x:
        
        root = Newton(Function,Derivative,i)
        
        if root != False:
            
            croot = np.round(root, tolerancia)
            
            if croot not in Roots:
                Roots = np.append(Roots,croot)
                
    Roots.sort()
    
    return Roots

print(Raices(x))


def muller(f,x0,x1,x2): 
    f1 = (f(1)-f(x0))/(x1-x0)
    f2 = (f(2)-f(1))/(x2-x1)
    a = (f2-f1)/((x1-x0)+(x2-x1))
    b = a * (x2-x1) + f2
    c = f(x2)

    res = 0
    res1 = (-2*c)/(b+np.sqrt(b**2-4*a*c))
    res2= (-2*c)/(b-np.sqrt(b**2-4*a*c))
    if b > 0: 
        res = res1
    else: 
        res = res2

    x3 = Newton

    i=0
    while i != 100:
        epsilon= f(x3) < 1*np.exp(-10)
        i+=1