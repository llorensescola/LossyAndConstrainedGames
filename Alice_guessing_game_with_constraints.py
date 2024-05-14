import cvxpy as cp
import numpy as np
import itertools as it
import time


#Number of repetitions
r=1

#Inputs and outputs
X=2**r
A=2**r
Y=2**r
B=2**r


#Local dimension of the referee
dim=2

D=dim**r

n0=1+X*A+Y*B
n=D*n0

def xa(x,a):
    return 1+x*A+a

def yb(y,b):
    return 1+X*A+y*B+b

from toqito.states import basis

# The basis: {|0>, |1>}:
e_0, e_1 = basis(2, 0), basis(2, 1)

# The basis: {|+>, |->}:
e_p = (e_0 + e_1) / np.sqrt(2)
e_m = (e_0 - e_1) / np.sqrt(2)

# The dimension of referee's measurement operators:
dim = 2

# Define the predicate matrix V(a,b|x,y) \in Pos(R)
V = np.zeros([dim, dim,A,B,X,Y])

# V(0,0|0,0) = |0><0|
V[:, :, 0, 0, 0, 0] = e_0 * e_0.conj().T
# V(1,1|0,0) = |1><1|
V[:, :, 1, 1, 0, 0] = e_1 * e_1.conj().T
# V(0,0|1,1) = |+><+|
V[:, :, 0, 0, 1, 1] = e_p * e_p.conj().T
# V(1,1|1,1) = |-><-|
V[:, :, 1, 1, 1, 1] = e_m * e_m.conj().T

# The probability matrix encode \pi(0,0) = \pi(1,1) = 1/2
prob_mat = 1/2*np.identity(2) 



# The probability matrix encode \pi(0,0) = \pi(1,1)  \pi(2,2) = 1/3
prob_mat = 1/X*np.identity(X)

def probwin(epsilon):
    M = cp.Variable((n,n), hermitian=True)
    
    cons=[M>>0]
    
    #0. Normalization
    cons+=[cp.trace(M[0::n0,0::n0])==1]
    
    
    #1.
    
    for (x,j) in it.product(np.arange(X),np.arange(n0)):
        cons+=[cp.sum([M[xa( x, a)::n0, j::n0] for a in range(A)])==M[0::n0,j::n0]]
        
    for (x,j) in it.product(np.arange(X),np.arange(n0)):
        cons+=[cp.sum([M[j::n0, xa( x, a)::n0] for a in range(A)])==M[j::n0,0::n0]]
        
    
    for (y,j) in it.product(np.arange(Y),np.arange(n0)):
        cons+=[cp.sum([M[yb(y,b)::n0,j::n0] for b in range(B)])==M[0::n0,j::n0]]
        
    for (y,j) in it.product(np.arange(Y),np.arange(n0)):
        cons+=[cp.sum([M[j::n0,yb(y,b)::n0] for b in range(B)])==M[j::n0,0::n0]]



    #2.
    
    for (x,a,c) in it.product(np.arange(X),np.arange(A),np.arange(A)):
        if a!=c:
            cons+=[M[xa(x,a)::n0,xa(x,c)::n0]==0]
    
    for (y,b,d) in it.product(np.arange(Y),np.arange(B),np.arange(B)):
        if b!=d:
            cons+=[M[yb(y,b)::n0,yb(y,d)::n0]==0]
        
    
    #3.
    for (x,a) in it.product(np.arange(X),np.arange(A)):
        cons+=[M[xa(x,a)::n0,xa(x,a)::n0]==M[0::n0,xa(x,a)::n0]]
        cons+=[M[xa(x,a)::n0,xa(x,a)::n0]==M[xa(x,a)::n0,0::n0]]
    
    for (y,b) in it.product(np.arange(Y),np.arange(B)):
        cons+=[M[yb(y,b)::n0,yb(y,b)::n0]==M[0::n0,yb(y,b)::n0]]
        cons+=[M[yb(y,b)::n0,yb(y,b)::n0]==M[yb(y,b)::n0,0::n0]]
    
    for (x,a,y,b) in it.product(np.arange(X),np.arange(A),np.arange(Y),np.arange(B)):
        cons+=[M[xa(x,a)::n0,yb(y,b)::n0]==M[yb(y,b)::n0,xa(x,a)::n0]]
    
    
    cons+=[cp.real(cp.sum([cp.trace(V[:, :,a ,a, x, x].conj().T@M[0::n0,yb(x,1-a)::n0]) for (x,a) in it.product(np.arange(X),np.arange(A))]))<=X*epsilon]
    
    
    p_win = cp.Constant(0) 
    for (a,x) in it.product(np.arange(A),np.arange(X)):
        p_win+=prob_mat[x,x]*cp.trace(V[:, :,a ,a, x, x].conj().T@M[xa(x,a)::n0,0::n0])
        
    
    objective = cp.Maximize(cp.real(p_win))
    problem = cp.Problem(objective, cons)
    start_time_SDP = time.time()
    cs_val = problem.solve()
    # Record the end time
    end_time_SDP = time.time()
    
    # Calculate the elapsed time
    elapsed_time_SDP = end_time_SDP - start_time_SDP
    
    # Print the elapsed time
    #print(f"Elapsed time for the SDP: {elapsed_time_SDP} seconds")
    return cs_val




