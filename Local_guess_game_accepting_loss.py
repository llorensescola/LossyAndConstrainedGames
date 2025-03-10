import cvxpy as cp
import numpy as np
import itertools as it



#Number of repetitions
r=2

#Inputs and outputs
X=2
A=3**r
Y=2
B=3**r



#Local dimension of the referee
dim=2

D=dim**r

n0=1+X*A+Y*B
n=D*n0

def xa(x,a):
    return 1+x*A+a

def yb(y,b):
    return 1+X*A+y*B+b


# The basis: {|0>, |1>}:
e_0, e_1 = np.array([[1],[0]]), np.array([[0],[1]])
# The basis: {|+>, |->}:
e_p = (e_0 + e_1) / np.sqrt(2)
e_m = (e_0 - e_1) / np.sqrt(2)


# The dimension of referee's measurement operators:
dim = 2
# The number of outputs for Alice and Bob:
a_out, b_out = 3, 3
# The number of inputs for Alice and Bob:
a_in, b_in = 2, 2

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

def r_binary(num):
    # if 0 <= num < D:  # Ensure the number is in the range [0, D-1]
    binary_str = bin(num)[2:]  # Convert to binary and remove the '0b' prefix
    binary_str = binary_str.zfill(r)  # Ensure it's r bits by adding leading zeros if needed
    return binary_str
    # else:
    #     return "Invalid input"
def ternary(n):
    # Helper function to convert decimal to trinary
    if n == 0:
        return "0t0"
    digits = []
    while n:
        digits.append(str(n % 3))
        n //= 3
    return "0t" + ''.join(digits[::-1])

def r_trinary(num):
    # if 0 <= num <= A:  # Ensure the number is in the range [0, 2]
    trinary_str = ternary(num)[2:]  # Convert to trinary and remove the '0t' prefix
    trinary_str = trinary_str.zfill(r)  # Ensure it's r trits by adding leading zeros if needed
    return trinary_str
    # else:
    #     return "Invalid input"

#Hamming wheight of 2's 
def h2(b):
    numberof2=0
    for i in range(r):
        if r_trinary(b)[i]=='2':
            numberof2+=1
    return numberof2


def pred_i(a,b,x,z):
    if z==x:
        if b==2:
            return 1
        if a==b:
            return 1
        if a==1-b:
            return 0
    if z!=x:
        return 1
    
    
    
def accept(v,b,x,z,upperbound,lowerbound):
    #a \in {0,1}^n
    #b \in {0,1,2}^n
    #x \in {0,1}
    #z \in {0,1}^n
    if h2(b)>upperbound:
        #print('Too many no loss')
        return 0
    if h2(b)<lowerbound:
        #print('Too few no loss')
        return 0
    else:
        #print('Good number of no loss')
        prod=1
        for i in range(r):
            prod*=pred_i(int(r_binary(v)[i]),int(r_trinary(b)[i]),x,int(r_binary(z)[i]))
        return prod

def Vmat(z,a):
    Vza=V[:,:,int(r_binary(a)[0]),int(r_binary(a)[0]),int(r_binary(z)[0]),int(r_binary(z)[0])]
    for i in range(1,r):
            Vza=np.kron(Vza,V[:,:,int(r_binary(a)[i]),int(r_binary(a)[i]),int(r_binary(z)[i]),int(r_binary(z)[i])])
    return Vza

def probwin(ubound,lbound): #We no longer need eta
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
    
  
    
    p_win = cp.Constant(0) 
    for (z,x,v,b) in it.product(np.arange(D),np.arange(X),np.arange(2**r),np.arange(B)):
        if accept(v,b,x,z,ubound,lbound)==1:
                p_win+=cp.trace(Vmat(z,v)@M[xa(x,b)::n0,yb(x,b)::n0])
        
    
    objective = cp.Maximize(cp.real(p_win))
    problem = cp.Problem(objective, cons)
    
    cs_val = problem.solve()
    
    return cs_val/D


    