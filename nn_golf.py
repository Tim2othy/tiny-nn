from numpy import dot,exp,max,random,sum,maximum,argmax,log
from load_mnist import xs,xt,ys,yt
D=dot
R=lambda x:maximum(0,x)
P=lambda x:(x>0)
def B(b,w,i,u):
    g=D(u,w.T)
    w-=0.04*D(i.T,u)
    b-=0.04*sum(u)
    return g
I=lambda a,b:random.rand(a,b)-0.5
w,c,v,d,x,j,t,e=(I(784,100),I(1,100),I(100,50),I(1,50),I(50,10),I(1,10),10000,0)
for i in range(t*6):
    q=R(D(xt[i],w)+c)
    n=R(D(q,v)+d)
    s=D(n,x)+j
    r=exp(s-max(s))/sum(exp(s-max(s)))
    B(c,w,xt[i],B(d,v,q,B(j,x,n,r-yt[i])*P(n))*P(q))
    e-=log(r[0,argmax(yt[i])])
    if (i+1)%t==0:
        print(f"At {i+1}/{t*6} the error is {e/t}")
        e=0
for i in range(t):
    e+=argmax(ys[i])!=argmax(D(R(D(R(D(xs[i],w)+c),v)+d),x)+j)
print(f"Test loss: {e/t}")
