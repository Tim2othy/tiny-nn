from numpy import dot,exp,max,mean,random,sum
from load_mnist import xs,xt,ys,yt
s = lambda x: exp(x - max(x)) / sum(exp(x - max(x)))
def b(b,w,i,u):
    g=dot(u,w.T)
    w-=0.04*dot(i.T,u)
    b-=0.04*sum(u)
    return g
f=lambda a,b:random.rand(a, b)-0.5
w,c,v,d,x,j,t,e=(f(784,100),f(1,100),f(100,50),f(1,50),f(50,10),f(1,10),10000,0)
for i in range(t*6):
    q=dot(xt[i],w)+c
    n=dot(q,v)+d
    r=s(dot(n,x)+j)
    b(c,w,xt[i],b(d,v,q,b(j,x,n,(r-yt[i])/r.size)))
    e+=mean((yt[i]-r)**2)
    if (i+1)%t==0:
        print(f"At {i+1}/{t*6} the error is {e/t}")
        e=0
for i in range(t):
    e+=mean((ys[i]-s(dot(dot(dot(xs[i],w)+c,v)+d,x)+j))**2)
print(f"Test loss: {e/t}")