#devtools::install_github("OHDSI/Cyclops")


p <- 100
n <- 1000
beta1 <- c(0.5,0,0,-1,1.2)
beta2 <- seq(0,0,length=p-length(beta1))
beta <- c(beta1,beta2)

x <- matrix(rnorm(p*n,mean=0,sd=1),ncol = p)
y <- NULL

for(i in 1:n){
    yi <- rbinom(1,1,exp(x[i,]%*%beta)/(1+exp(x[i,]%*%beta)))
    y <- c(y, yi)
}


cyclopsData <- createCyclopsData(y ~ x - 1,modelType = "lr")
abridge <- fitCyclopsModel(cyclopsData, prior = createAbridgePrior("bic"),
           control = createControl(noiseLevel = "silent"))


