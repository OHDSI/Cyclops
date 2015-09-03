library(glmnet)
glmnetCoxMM <- function(N, p, c, e, lambda = -1) {
    # N = number of patients
    # p = number of covariates
    # c = censoring probability
    # e = distance between consecutive Poisson regressions to stop
    # lambda = lambda values to use; leave it alone for now because glmnet handles lambda weird
    x=matrix(rnorm(N*p),N,p)
    beta=1:p/p
    fx=x%*%beta
    hx=exp(fx)
    ty=rexp(N,hx)
    tcens=1 - rbinom(n=N,prob=c,size=1)# censoring indicator
    y=cbind(time=ty,status=tcens) # y=Surv(ty,1-tcens) with library(survival)

    fitCox = glmnet(x, y, family = "cox")
    iterCox = dim(fitCox$beta)[2]
    betaCox = fitCox$beta[,iterCox]

    x = x[order(y[,1]),]
    y = y[order(y[,1]),]
    ci = y[,2]

    iterCounts = c()
    iterBeta = rep(0, dim(x)[2])
    iterPoisson = 0
    betaLast = rep(0, dim(x)[2])
    betaNext = rep(0, dim(x)[2])
    while((dist(rbind(betaLast, betaNext)) > e) | (iterPoisson == 0)) {
        iterPoisson = iterPoisson + 1
        betaLast = betaNext
        tj = exp(x%*%betaLast)
        si = sapply(1:length(tj), FUN = function(k){return(sum(tj[k:length(tj)]))})
        ki = sapply(1:length(tj), FUN = function(k){return(sum(ci[1:k] / si[1:k]))})
        wi = ci * ki

        kZero = which(ki==0)
        wZero = wi[kZero]
        if (!all(wZero==0)) stop("ki zero but wi not zero")
        ki[kZero] = 0.01

        if (lambda == -1) {
            fitPoisson = glmnet(x, 1 / ki, family = "poisson", weights = wi)
        } else {
            fitPoisson = glmnet(x, 1 / ki, family = "poisson", weights = wi, lambda = lambda)
        }
        t = dim(fitPoisson$beta)[2]
        betaNext = fitPoisson$beta[,t]
        iterBeta = cbind(iterBeta, betaNext)

        distBetaNew = apply(fitPoisson$beta, MARGIN = 2, FUN = function(x){return(dist(rbind(x,0)))})
        distBetaOld = dist(rbind(betaLast, 0))

        iterCounts = c(iterCounts, t - which(distBetaNew > distBetaOld)[1])
    }
    colnames(iterBeta) = 0:iterPoisson

    result = list(beta = beta, betaCox = betaCox, betaPoisson = betaNext,
                  iterCox = iterCox, iterPoisson = iterPoisson,
                  #iterBeta = iterBeta,
                  iterCounts = iterCounts, iterCountsTotal = sum(iterCounts))
    return(result)
}
