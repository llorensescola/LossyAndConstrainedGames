import cvxpy as cp
import numpy as np
import itertools as it



#Inputs and outputs
X=2
A=2
Y=2
B=2


#Local dimension of the referee
dim=2

D=dim

n0=1+X*A+Y*B
n=D*n0

def xa(x,a):
    return 1+x*A+a

def yb(y,b):
    return 1+X*A+y*B+b




e_0, e_1 = np.array([[1],[0]]), np.array([[0],[1]])
e_p = (e_0 + e_1) / np.sqrt(2)
e_m = (e_0 - e_1) / np.sqrt(2)


# Define the predicate matrix V(a,b|x,y) \in Pos(R)
V = np.zeros([dim, dim,2,2,2,2])

# V(0,0|0,0) = |0><0|
V[:, :, 0, 0, 0, 0] = e_0 @ e_0.conj().T
# V(1,1|0,0) = |1><1|
V[:, :, 1, 1, 0, 0] = e_1 @ e_1.conj().T
# V(0,0|1,1) = |+><+|
V[:, :, 0, 0, 1, 1] = e_p @ e_p.conj().T
# V(1,1|1,1) = |-><-|
V[:, :, 1, 1, 1, 1] = e_m @ e_m.conj().T

# The probability matrix encode \pi(0,0) = \pi(1,1) = 1/2
prob_mat = 1/X*np.identity(X) 



def Vmat(z,a):
    return V[:, :,a ,a, z, z]
    
prob_mat = 1/X*np.identity(X) 

zero_mat=np.zeros([D,D])

#The following function returns the upper bound corresponding to the level k=`1+AB'on the optimal winning probability given the constraint imposed by the parameter epsilon


def probwin(epsilon):
    M = cp.Variable((n,n), hermitian=True)
    
    cons=[M>>0]
    
    cons+=[cp.trace(M[0::n0,0::n0])==1]
    
    
#1.--------------------------   
    for (x,j) in it.product(np.arange(X),np.arange(n0)):
        cons+=[M[0::n0,j::n0]>>cp.sum([M[xa( x, a)::n0, j::n0] for a in range(A)])]
        
    
    for (y,j) in it.product(np.arange(Y),np.arange(n0)):
        cons+=[M[0::n0,j::n0]>>cp.sum([M[yb(y,b)::n0,j::n0] for b in range(B)])]
    
    
   
    
    
#2.---------------------------------------   
    
    for (x,a,a1) in it.product(np.arange(X),np.arange(A),np.arange(A)):
        if a!=a1:
            cons+=[M[xa(x,a)::n0,xa(x,a1)::n0]==zero_mat]


            
    
    for (y,b,b1) in it.product(np.arange(Y),np.arange(B),np.arange(B)):
        if b!=b1:
            cons+=[M[yb(y,b)::n0,yb(y,b1)::n0]==zero_mat]
 
     
#3.-------------------------------------------      
    for (x,a) in it.product(np.arange(X),np.arange(A)):
        cons+=[M[xa(x,a)::n0,xa(x,a)::n0]==M[0::n0,xa(x,a)::n0]]
        cons+=[M[xa(x,a)::n0,xa(x,a)::n0]==M[xa(x,a)::n0,0::n0]]
    
    for (y,b) in it.product(np.arange(Y),np.arange(B)):
        cons+=[M[yb(y,b)::n0,yb(y,b)::n0]==M[0::n0,yb(y,b)::n0]]
        cons+=[M[yb(y,b)::n0,yb(y,b)::n0]==M[yb(y,b)::n0,0::n0]]
        
    for (x,a,y,b) in it.product(np.arange(X),np.arange(A),np.arange(Y),np.arange(B)):
        cons+=[M[xa(x,a)::n0,yb(y,b)::n0]==M[yb(y,b)::n0,xa(x,a)::n0]]
        
        

    
    
    cons+=[cp.real(cp.sum([cp.trace(M[xa(x,a)::n0,yb(x,1-a)::n0]) for a in range(A) for x in range(X)]))>=X*epsilon]
    
    
    p_win = cp.Constant(0) 
    for (a,x) in it.product(np.arange(A),np.arange(X)):
        p_win+=cp.trace(Vmat(x,a).conj().T@M[xa(x,a)::n0,yb(x,a)::n0])
    
    objective = cp.Maximize(cp.real(p_win))
    problem = cp.Problem(objective, cons)
    
    cs_val = problem.solve()

    return cs_val/X

   



