import numpy as n

from load_mnist import xs, xt, ys, yt


def p(yt,yp): return 2*(yp-yt)/yt.size
def s(x):
    h=n.exp(x-n.max(x))
    return h / n.sum(h)
def b(b,w,i,u):
    g=n.dot(u,w.T)
    w-=0.04*n.dot(i.T,u)
    b-=0.04*n.sum(u)
    return g
def f(a,b):return n.random.rand(a,b)-0.5
w1,b1,w2,b2,w3,b3=(f(784,100),f(1,100),f(100,50),f(1,50),f(50,10),f(1,10))
t=10^4
e=0
for i in range(t*6):
    o1=n.dot(xt[i],w1)+b1
    o2=n.dot(o1,w2)+b2
    r=s(n.dot(o2,w3)+b3)
    b(b1,w1,xt[i],b(b2,w2,o1,b(b3,w3,o2,p(yt[i],r))))
    e += n.mean(n.power(yt[i]-r,2))
    if (i+1) % t == 0:
        print(f"At {i+1}/{t*6} the error is {e/t:.3f}")
        e=0
for i in range(t):
    e+=n.mean(n.power(ys[i]-s(n.dot(n.dot(n.dot(xs[i],w1)+b1,w2)+b2,w3)+b3),2))
print(f"Test loss: {e / t:.3f}")
