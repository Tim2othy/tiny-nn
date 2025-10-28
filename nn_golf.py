from numpy import dot,exp,max,mean,random,sum
from load_mnist import xs,xt,ys,yt
def s(x):
    h=exp(x-max(x))
    return h/sum(h)
def b(b,w,i,u):
    g=dot(u,w.T)
    w-=0.04*dot(i.T,u)
    b-=0.04*sum(u)
    return g
def f(a,b):return random.rand(a,b)-0.5
w1,b1,w2,b2,w3,b3,t,e=(f(784,100),f(1,100),f(100,50),f(1,50),f(50,10),f(1,10),10000,0)
for i in range(t*6):
    o1=dot(xt[i],w1)+b1
    o2=dot(o1,w2)+b2
    r=s(dot(o2,w3)+b3)
    b(b1,w1,xt[i],b(b2,w2,o1,b(b3,w3,o2,(r-yt[i])/r.size)))
    e+=mean((yt[i]-r)**2)
    if (i+1)%t==0:
        print(f"At {i+1}/{t*6} the error is {e/t}")
        e=0
for i in range(t):
    e+=mean((ys[i]-s(dot(dot(dot(xs[i],w1)+b1,w2)+b2,w3)+b3))**2)
print(f"Test loss: {e/t}")