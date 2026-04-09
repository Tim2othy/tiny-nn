from numpy import dot,exp,max,random,sum,argmax,log
from load_mnist import xs,xt,ys,yt
D=dot
R=lambda x:x*(x>0)
def B(b,w,i,e):
 g=D(e,w.T)
 w-=.02*D(i.T,e)
 b-=.02*sum(e)
 return g
I=lambda x,y:random.rand(x,y)-.5
u,v,w,a,b,c,t,e=(I(784,99),I(99,50),I(50,10),I(1,99),I(1,50),I(1,10),10000,0)
for i in range(t*6):
 r=exp((m:=(s:=D(n:=R(D(q:=R(D(xt[i],u)+a),v)+b),w)+c)-max(s)))/sum(exp(m))
 B(a,u,xt[i],(q>0)*B(b,v,q,(n>0)*B(c,w,n,r-yt[i])))
 e-=log(r[0,argmax(yt[i])])
 if-~i%t<1:
  print(f"At {i+1}/{t*6} the error is {e/t}")
  e=0
for i in range(t):
 e+=argmax(ys[i])!=argmax(D(R(D(R(D(xs[i],u)+a),v)+b),w)+c)
print(f"Test loss: {e/t}")
