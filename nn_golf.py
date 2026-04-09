from numpy import dot,exp,max,random,sum,maximum,argmax,log
from load_mnist import xs,xt,ys,yt
D=dot
R=lambda x:maximum(0,x)
def B(b,w,i,u):
    g=D(u,w.T)
    w-=0.04*D(i.T,u)
    b-=0.04*sum(u)
    return g
I=lambda a,b:random.rand(a,b)-0.5
u,v,w,a,b,c,t,e=(I(784,99),I(99,50),I(50,10),I(1,99),I(1,50),I(1,10),10000,0)
for i in range(t*6):
    q=R(D(xt[i],u)+a)
    n=R(D(q,v)+b)
    s=D(n,w)+c
    r=exp(s-max(s))/sum(exp(s-max(s)))
    B(a,u,xt[i],B(b,v,q,B(c,w,n,r-yt[i])*(n>0))*(q>0))
    e-=log(r[0,argmax(yt[i])])
    if (i+1)%t==0:
        print(f"At {i+1}/{t*6} the error is {e/t}")
        e=0
for i in range(t):
    e+=argmax(ys[i])!=argmax(D(R(D(R(D(xs[i],u)+a),v)+b),w)+c)
print(f"Test loss: {e/t}")
