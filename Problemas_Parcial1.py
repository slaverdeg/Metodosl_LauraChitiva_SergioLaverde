import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
sym.init_printing(use_unicode=True)

def Function(x):
    return ((3*x**5)+(5*x**4)-(x**3))

def Derivative(f,x,h=1e-6):
    return (f(x+h)-f(x-h))/(2*h)

def GetNewtonMethod(f,df,xn,itmax=100,precision=1e-10):
    
    error = 1.
    it = 0
    
    while error > precision and it < itmax:
        
        try:
            
            xn1 = xn - f(xn)/(df(f,xn))
            # Criterio de parada
            error = np.abs(f(xn)/(df(f,xn)))
            
        except ZeroDivisionError:
            print('Division por cero')
            
        xn = xn1
        it += 1
    
    if it == itmax:
        return False
    else:
        return xn
    

def GetAllRoots(x, tolerancia=10):
    
    Roots = np.array([])
    
    for i in x:
        
        root = GetNewtonMethod(Function,Derivative,i)
        
        if root != False:
            
            croot = np.round(root, tolerancia)
            
            if croot not in Roots:
                Roots = np.append(Roots,croot)
                
    Roots.sort()
    
    return Roots

x = np.linspace(-2,1,100)
#Roots_df = GetAllRoots(x)
#Roots_df

f = open('./Parabolico.csv')
lineas = f.readlines()
f.close()
x = []
y = []
for k in lineas[1:]:
    #print(k.strip().split(';'))
    x.append(float(k.strip().split(',')[0]))
    y.append(float(k.strip().split(',')[1]))
#print(x,y)

X = np.array(x)
Y = np.array(y)
#print(x,y)

#plt.scatter(x,y)
vx = x[1]-x[0]

def Lagrange(x,X,i):
    
    L = 1
    
    for j in range(X.shape[0]):
        if i != j:
            L *= (x - X[j])/(X[i]-X[j])
            
    return L

def Interpolate(x,X,Y):
    
    Poly = 0
    
    for i in range(X.shape[0]):
        Poly += Lagrange(x,X,i)*Y[i]
        
    return Poly

x = np.linspace(1.4,5.6,100)
y = Interpolate(x,X,Y)

plt.plot(x,y,color='k')
plt.scatter(X,Y,color='r',marker='o')
plt.show()

_x = sym.Symbol('x',real=True)
y = Interpolate(_x,X,Y)
f = sym.simplify(y)

def Function(x):
    return x*(0.363970234266202-(0.0554912422401579*x))

def Derivative(f,x,h=1e-6):
    return (f(x+h)-f(x-h))/(2*h)

def Second_Derivative(f,x,h=1e-6,h2=1e-6):
    return ((f(x+2*h)-2*f(x)+f(x-2*h))/(4*h2**2))

def GetNewtonMethod(f,df,xn,itmax=100,precision=1e-8):
    
    error = 1.
    it = 0
    
    while error > precision and it < itmax:
        
        try:
            
            xn1 = xn - f(xn)/(df(f,xn))
            # Criterio de parada
            error = np.abs(f(xn)/(df(f,xn)))
            
        except ZeroDivisionError:
            print('Division por cero')
            
        xn = xn1
        it += 1
    
    if it == itmax:
        return False
    else:
        return xn
    
def GetNewtonMethod_df(f,fd,xn,itmax=100,precision=1e-8):
    
    error = 1.
    it = 0
    
    while error > precision and it < itmax:
        
        try:
            
            xn1 = xn - fd(f,xn)/(-0.111)
            # Criterio de parada
            error = np.abs(-0.111)
            
        except ZeroDivisionError:
            print('Division por cero')
            
        xn = xn1
        it += 1
    
    if it == itmax:
        return False
    else:
        return xn

def GetAllRoots(x, tolerancia=10):
    
    Roots = np.array([])
    
    for i in x:
        
        root = GetNewtonMethod(Function,Derivative,i)
        
        if root != False:
            
            croot = np.round(root, tolerancia)
            
            if croot not in Roots:
                Roots = np.append(Roots,croot)
                
    Roots.sort()
    
    return Roots

def GetAllRoots_df(x, tolerancia=10):
    
    Roots = np.array([])
    
    for i in x:
        
        root = GetNewtonMethod_df(Function,Derivative,i)
        
        if root != False:
            
            croot = np.round(root, tolerancia)
            
            if croot not in Roots:
                Roots = np.append(Roots,croot)
                
    Roots.sort()
    
    return Roots

x = np.linspace(0,7,100)

Roots = GetAllRoots(x)
Roots

#Roots_df = GetAllRoots(x)
#Roots_df

y = Interpolate(x,X,Y)
y2 = Derivative(Function,x)
y3 = Second_Derivative(Function,x)

plt.plot(x,y,color='k')
plt.plot(x,y2,color='k')
plt.plot(x,y3,color='r')
plt.plot(x,np.tile(0,len(x)))
plt.show()

