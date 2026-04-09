from numpy import dot,exp,max,random,sum,argmax,log
from load_mnist import xs,xt,ys,yt
p=dot
def b(b,w,i,u):
    g=p(u,w.T)
    w-=0.04*p(i.T,u)
    b-=0.04*sum(u)
    return g
f=lambda a,b:random.rand(a, b)-0.5
w,c,v,d,x,j,t,e=(f(784,100),f(1,100),f(100,50),f(1,50),f(50,10),f(1,10),10000,0)
for i in range(t*6):
    q=p(xt[i],w)+c
    n=p(q,v)+d
    s=p(n,x)+j
    r=exp(s-max(s))/sum(exp(s-max(s)))
    b(c,w,xt[i],b(d,v,q,b(j,x,n,(r-yt[i])/10)))
    e-=log(r[0,argmax(yt[i])])
    if (i+1)%t==0:
        print(f"At {i+1}/{t*6} the error is {e/t}")
        e=0
for i in range(t):
    e+=argmax(ys[i])!=argmax(p(p(p(xs[i],w)+c,v)+d,x)+j)
print(f"Test loss: {e/t}")