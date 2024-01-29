
import numpy as np
def Wrapper(x, func):
## Schwefel's function. VarMin, VarMax = -500   , 500
  m, n = np.shape(x)
  f = np.zeros(n)
  for j in range(0, n):
    f[j] = func(x[0:m,j])
  return f

def Schwefel(x):
## Schwefel's function. VarMin, VarMax = -500   , 500
  m, n = np.shape(x)
  f = np.zeros(n)
  for j in range(0, n):
    f[j] = 418.982887*m-np.sum(x[0:m,j]*np.sin(np.sqrt(abs(x[0:m,j]))))
  return f
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Rosenbrock(x):
## Rosenbrock's valley. VarMin, VarMax = -2.048 , 2.048
  m, n = np.shape(x)
  f = np.zeros(n)
  for j in range(0,n):
    f[j] = np.sum((1.0-x[0:m,j])**2) \
         + np.sum((x[1:m,j]-x[0:m-1,j])**2)
  return f
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Rastrigin(x):
## Rastrigin's function. VarMin, VarMax = -5.12  , 5.12
  m, n = np.shape(x)
  f = np.zeros ( n )
  for j in range(0,n):
    f[j] = 10.0 * float(m)
    for i in range(0,m):
      f[j] = f[j]+x[i,j]**2-10.0*np.cos(2.0*np.pi*x[i,j])
  return f
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Griewank(x):
## Griewank's function. VarMin, VarMax = -600   , 600
  m, n = np.shape(x)
  f = np.zeros(n)
  y = list(range(1, m+1))
  y[0:m] = np.sqrt(y[0:m])
  for j in range(0,n):
    f[j] = np.sum(x[0:m,j]**2) / 4000.0 \
      - np.prod(np.cos(x[0:m,j] / y[0:m]))+1.0
  return f
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Ackley(x):
## Ackley's function. VarMin, VarMax = -1     , 1
  m, n = np.shape(x)
  f = np.zeros(n)
  a = 20.0
  b = 0.2
  c = 0.2
  for j in range(0, n):
    f[j] = - a * np.exp(-b*np.sqrt(np.sum(x[0:m,j]**2) / float(m))) \
      - np.exp(np.sum(np.cos(c*np.pi*x[0:m,j])) / float(m)) \
      + a + np.exp(1.0)
  return f