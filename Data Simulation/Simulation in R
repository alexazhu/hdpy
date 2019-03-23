#simulation start#

p=2200
n=200
b1 <- c(5,rep(5/sqrt(10),10),-5,rep(-5/sqrt(10),10),3,rep(3/sqrt(10),10),-3,rep(-3/sqrt(10),10),rep(0,p-44))
b2 <- c(5,rep(-5/sqrt(10),3),rep(5/sqrt(10),7),-5,rep(5/sqrt(10),3),rep(-5/sqrt(10),7),3,rep(-3/sqrt(10),3),rep(3/sqrt(10),7),-3,rep(3/sqrt(10),3),rep(-3/sqrt(10),7),rep(0,p-44))
b3 <- c(5,rep(5/10,10),-5,rep(-5/10,10),3,rep(3/10,10),-3,rep(-3/10,10),rep(0,p-44))
b4 <- c(5,rep(-5/10,3),rep(5/10,7),-5,rep(5/10,3),rep(-5/10,7),3,rep(-3/10,3),rep(3/10,7),-3,rep(3/10,3),rep(-3/10,7),rep(0,p-44))


#simulation function#
simulation <- function(result,b,datanum){
for(q in 1:50){
  set.seed(q)
e <-rnorm(n,mean = 0,sd=(sum(b^2))/4)
gene <- vector()
Xtf <- matrix(rep(0,n*200),ncol=200)
X <- matrix(nrow=n,ncol=p)
for (i in 1:n) {
  Xtf[i,] <- rnorm(200)
  for(j in 1:200){
    gene <- rnorm(n=10,mean=0.7*Xtf[i,j],0.51)
    X[i,((j-1)*10+j):(j*10+j)] <- c(Xtf[i,j],gene)
  }
}
Y <- vector()
for (i in 1:n) {
  Y[i] <- X[i,]%*%b+e[i]
}
