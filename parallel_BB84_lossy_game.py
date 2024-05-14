import cvxpy as cp
import numpy as np
import itertools as it
import time
import math

#Number of repetitions
r=2

#Inputs and outputs
X=2**r
A=3**r-1
Y=2**r
B=3**r-1


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


e_0, e_1 = basis(2, 0), basis(2, 1)
e_p = (e_0 + e_1) / np.sqrt(2)
e_m = (e_0 - e_1) / np.sqrt(2)


# Define the predicate matrix V(a,b|x,y) \in Pos(R)
V = np.zeros([dim, dim,2,2,2,2])

# V(0,0|0,0) = |0><0|
V[:, :, 0, 0, 0, 0] = e_0 * e_0.conj().T
# V(1,1|0,0) = |1><1|
V[:, :, 1, 1, 0, 0] = e_1 * e_1.conj().T
# V(0,0|1,1) = |+><+|
V[:, :, 0, 0, 1, 1] = e_p * e_p.conj().T
# V(1,1|1,1) = |-><-|
V[:, :, 1, 1, 1, 1] = e_m * e_m.conj().T

# The probability matrix encode \pi(0,0) = \pi(1,1) = 1/2
prob_mat = 1/X*np.identity(X) 

zero_mat=np.zeros([D,D])

def probwin(eta):
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
        
    
    cons+=[cp.sum([cp.trace(M[xa(x,2)::n0,yb(x,2)::n0]) for x in np.arange(X)])==X*eta*(1-eta)/2]
    cons+=[cp.sum([cp.trace(M[xa(x,5)::n0,yb(x,5)::n0]) for x in np.arange(X)])==X*eta*(1-eta)/2]
    cons+=[cp.sum([cp.trace(M[xa(x,6)::n0,yb(x,6)::n0]) for x in np.arange(X)])==X*eta*(1-eta)/2]
    cons+=[cp.sum([cp.trace(M[xa(x,7)::n0,yb(x,7)::n0]) for x in np.arange(X)])==X*eta*(1-eta)/2]
    
    cons+=[
        cp.sum([
            1+
            cp.trace(cp.sum([M[xa(x,a)::n0,yb(x,b)::n0] for a in range(A) for b in range(B)]))
            -cp.trace(cp.sum([M[xa(x,a)::n0,0::n0] for a in range(A)]))
            -cp.trace(cp.sum([M[0::n0,yb(x,b)::n0] for  b in range(B)]))
            for x in range(X)])==X*(1-eta)**2
        ]
    
    for (x,a,b) in it.product(np.arange(X),np.arange(A),np.arange(B)):
        if a!=b:
            cons+=[M[xa(x,a)::n0,xa(x,b)::n0]==zero_mat]
    
    p_win = cp.Constant(0) 
    for (a,x) in it.product(np.arange(A),np.arange(X)):
        if math.floor(a/3) in [0,1] and a%3 in [0,1]:
            x0=math.floor(x/2)
            x1=x%2
            a0=math.floor(a/3)
            a1=a%3
            p_win+=cp.trace((np.kron(V[:, :,a0 ,a0, x0, x0],V[:, :,a1 ,a1, x1, x1])).conj()@M[xa(x,a)::n0,yb(x,a)::n0])

        
    
    
    objective = cp.Maximize(cp.real(p_win))
    problem = cp.Problem(objective, cons)
    start_time_SDP = time.time()
    cs_val = problem.solve()
    # Record the end time
    end_time_SDP = time.time()
    
    # Calculate the elapsed time
    elapsed_time_SDP = end_time_SDP - start_time_SDP
    # Print the elapsed time
    print(f"Elapsed time for the SDP: {elapsed_time_SDP} seconds")
    return cs_val/X
print('Forbiden ans: Parallel repetition QPV_BB84 level 1 with less outputs and simple constraints')
r=[]
for eta in np.arange(0.525,1,0.025):
    sol=probwin(eta)
    print(sol)
    r.append(sol)



