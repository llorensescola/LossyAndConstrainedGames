import cvxpy as cp
import numpy as np
import itertools as it



#Number of repetitions
r=1

#Inputs and outputs
X=3**r
A=3**r-1
Y=3**r
B=3**r-1


#Local dimension of the referee
dim=2

D=dim**r

n0=1+X*A+Y*B+X*A*Y*B
n=D*n0

def xa(x,a):
    return 1+x*A+a

def yb(y,b):
    return 1+X*A+y*B+b

def xayb(x,a,y,b):
    return 1+X*A+Y*B+x*A*Y*B+a*Y*B+y*B+b



import cmath
I=cmath.sqrt(-1)

# The basis: {|0>, |1>}:
e_0, e_1 = np.array([[1],[0]]), np.array([[0],[1]])

# The basis: {|+>, |->}:
e_p = (e_0 + e_1) / np.sqrt(2)
e_m = (e_0 - e_1) / np.sqrt(2)

# The basis: {|i>, |-i>}:
e_ip=(e_0 + I*e_1) / np.sqrt(2)
e_im=(e_0 - I*e_1) / np.sqrt(2)

# The dimension of referee's measurement operators:
dim = 2
# The number of outputs for Alice and Bob:
a_out, b_out = 3, 3
# The number of inputs for Alice and Bob:
a_in, b_in = 3, 3

# Define the predicate matrix V(a,b|x,y) \in Pos(R)

#LE: I add complex things
V = np.zeros([dim, dim, a_out, b_out, a_in, b_in],dtype = 'complex_')

# V(0,0|0,0) = |0><0|
V[:, :, 0, 0, 0, 0] = e_0 @ e_0.conj().T
# V(1,1|0,0) = |1><1|
V[:, :, 1, 1, 0, 0] = e_1 @ e_1.conj().T
# V(0,0|1,1) = |+><+|
V[:, :, 0, 0, 1, 1] = e_p @ e_p.conj().T
# V(1,1|1,1) = |-><-|
V[:, :, 1, 1, 1, 1] = e_m @ e_m.conj().T
# V(0,0|2,2) = |i><i|
V[:, :, 0, 0, 2, 2] =np.array([[0.5 , -0.5*I], [0.5*I, 0.5 ]])
# V(1,1|2,2) = |-i><-i|
V[:, :, 1, 1, 2, 2] =np.array([[0.5 , 0.5*I],[ -0.5*I, 0.5]])


# The probability matrix encode \pi(0,0) = \pi(1,1)  \pi(2,2) = 1/3

prob_mat = 1/X*np.identity(X) 

zero_mat=np.zeros([D,D])

#The following function returns the upper bound corresponding to the level k=`1+AB' on the optimal winning probability given the value of the lossy parameter eta

def probwin(eta):
    M = cp.Variable((n,n), hermitian=True)
    
    cons=[M>>0]
    
    cons+=[cp.trace(M[0::n0,0::n0])==1]
    
    
#1.--------------------------   
    for (x,j) in it.product(np.arange(X),np.arange(n0)):
        cons+=[M[0::n0,j::n0]>>cp.sum([M[xa( x, a)::n0, j::n0] for a in range(A)])]
        
    
    for (y,j) in it.product(np.arange(Y),np.arange(n0)):
        cons+=[M[0::n0,j::n0]>>cp.sum([M[yb(y,b)::n0,j::n0] for b in range(B)])]
    
    
    for (x,y,b,j) in it.product(np.arange(X),np.arange(Y),np.arange(B),np.arange(n0)):
        cons+=[M[yb(y,b)::n0,j::n0]>>cp.sum([M[xayb( x, a, y, b)::n0, j::n0] for a in range(A)])]
        
        
    # for (x,y,b,j) in it.product(np.arange(X),np.arange(Y),np.arange(B),np.arange(n0)):
    #     cons+=[cp.sum([M[xayb(x, a, y, b)::n0, j::n0] for a in range(A)])<<M[yb(y,b)::n0,j::n0]]    
    
    for (x,a,y,j) in it.product(np.arange(X),np.arange(A),np.arange(Y),np.arange(n0)):
          cons+=[cp.sum([M[xayb(x,a,y,b)::n0,j::n0] for b in range(B)])<<M[xa(x,a)::n0,j::n0]]
    
    
#2.---------------------------------------   
    
    for (x,a,a1) in it.product(np.arange(X),np.arange(A),np.arange(A)):
        if a!=a1:
            cons+=[M[xa(x,a)::n0,xa(x,a1)::n0]==zero_mat]

    for (x,a,a1,y,b) in it.product(np.arange(X),np.arange(A),np.arange(A),np.arange(Y),np.arange(B)):
        if a!=a1:
            cons+=[M[xa(x,a)::n0,xayb(x,a1,y,b)::n0]==zero_mat]
            
    for (x,a,a1,y,b) in it.product(np.arange(X),np.arange(A),np.arange(A),np.arange(Y),np.arange(B)):
        if a!=a1:
            cons+=[M[xayb(x,a,y,b)::n0,xa(x,a1)::n0]==zero_mat]      
    
    for (x,a,a1,y,b,y1,b1) in it.product(np.arange(X),np.arange(A),np.arange(A),np.arange(Y),np.arange(B),np.arange(Y),np.arange(B)):
        if a!=a1:
            cons+=[M[xayb(x,a,y,b)::n0,xayb(x,a1,y1,b1)::n0]==zero_mat]
            
            
            
    
    for (y,b,b1) in it.product(np.arange(Y),np.arange(B),np.arange(B)):
        if b!=b1:
            cons+=[M[yb(y,b)::n0,yb(y,b1)::n0]==zero_mat]
    for (x,a,y,b,b1) in it.product(np.arange(X),np.arange(A),np.arange(Y),np.arange(B),np.arange(B)):
        if b!=b1:
            cons+=[M[yb(y,b)::n0,xayb(x,a,y,b1)::n0]==zero_mat]
    for (x,a,y,b,b1) in it.product(np.arange(X),np.arange(A),np.arange(Y),np.arange(B),np.arange(B)):
        if b!=b1:
            cons+=[M[xayb(x,a,y,b)::n0,yb(y,b1)::n0]==zero_mat]
    for (x,a,x1,a1,y,b,b1) in it.product(np.arange(X),np.arange(A),np.arange(X),np.arange(A),np.arange(Y),np.arange(B),np.arange(B)):
        if b!=b1:
            cons+=[M[xayb(x,a,y,b)::n0,xayb(x1,a1,y,b1)::n0]==zero_mat]
     
#3.-------------------------------------------      
    for (x,a) in it.product(np.arange(X),np.arange(A)):
        cons+=[M[xa(x,a)::n0,xa(x,a)::n0]==M[0::n0,xa(x,a)::n0]]
        cons+=[M[xa(x,a)::n0,xa(x,a)::n0]==M[xa(x,a)::n0,0::n0]]
    
    for (y,b) in it.product(np.arange(Y),np.arange(B)):
        cons+=[M[yb(y,b)::n0,yb(y,b)::n0]==M[0::n0,yb(y,b)::n0]]
        cons+=[M[yb(y,b)::n0,yb(y,b)::n0]==M[yb(y,b)::n0,0::n0]]
        
    for (x,a,y,b) in it.product(np.arange(X),np.arange(A),np.arange(Y),np.arange(B)):
        cons+=[M[xa(x,a)::n0,yb(y,b)::n0]==M[yb(y,b)::n0,xa(x,a)::n0]]
        cons+=[M[xa(x,a)::n0,yb(y,b)::n0]==M[0::n0,xayb(x,a,y,b)::n0]]
        cons+=[M[xa(x,a)::n0,yb(y,b)::n0]==M[xayb(x,a,y,b)::n0,0::n0]]
        cons+=[M[xa(x,a)::n0,yb(y,b)::n0]==M[xayb(x,a,y,b)::n0,xa(x,a)::n0]]
        cons+=[M[xa(x,a)::n0,yb(y,b)::n0]==M[xa(x,a)::n0,xayb(x,a,y,b)::n0]]
        cons+=[M[xa(x,a)::n0,yb(y,b)::n0]==M[yb(y,b)::n0,xayb(x,a,y,b)::n0]]
        cons+=[M[xa(x,a)::n0,yb(y,b)::n0]==M[xayb(x,a,y,b)::n0,yb(y,b)::n0]]
        cons+=[M[xa(x,a)::n0,yb(y,b)::n0]==M[xayb(x,a,y,b)::n0,xayb(x,a,y,b)::n0]]
    
    

    
    cons+=[
        cp.sum([
            1+
            cp.trace(cp.sum([M[xa(x,a)::n0,yb(x,b)::n0] for a in range(A) for b in range(B)]))
            -cp.trace(cp.sum([M[xa(x,a)::n0,0::n0] for a in range(A)]))
            -cp.trace(cp.sum([M[0::n0,yb(x,b)::n0] for  b in range(B)]))
            for x in range(X)])==X*(1-eta)
        ]
    
    
    p_win = cp.Constant(0) 
    for (x,a) in it.product(np.arange(X),np.arange(A)):
        if a!=2:
            p_win+=cp.trace(V[:, :,a ,a, x, x]@M[xa(x,a)::n0,yb(x,a)::n0])
        
    
    
    objective = cp.Maximize(cp.real(p_win))
    problem = cp.Problem(objective, cons)
    
    cs_val = problem.solve()
    
    return cs_val/X

