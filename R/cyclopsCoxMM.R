#' Simulates data to test cox MM
#'
#' @description
#' Simulates data to test the cox MM algorithm.
#'
#' @param N number of patients
#' @param p number of covariates
#' @param c censoring probability
#' @param e distance between Poisson regressions to stop
#' @param intercept whether to include intercept in model, default = TRUE
#' @param beta vector to generate data from
#'
#' @return
#' Returns a lit with the following: \describe{
#' \item{seedCoefficients}{This is equal to input beta}
#' \item{goldCox}{Cox regression model from using survival package function coxph}
#' \item{cyclopsCox}{Cox regression model from using cyclops}
#' \item{goldPoisson}{Cox MM poisson model from using glm}
#' \item{cyclopsPoisson}{Cox MM poisson model from using cyclops} }
#'
#' @export
cyclopsCoxMM <- function(N, p, c, e, intercept = TRUE, beta = NA) {
    # N = number of patients
    # p = number of covariates
    # c = censoring probability
    # e = distance between consecutive Poisson regressions to stop
    # intercept = whether to include intercept in model
    # beta = beta vector to generate data from
    x=matrix(rnorm(N*p),N,p)
    if (is.na(beta)) beta=1:p/p
    fx=x%*%beta
    hx=exp(fx)
    ty=rexp(N,hx)
    tcens=1 - rbinom(n=N,prob=c,size=1)# censoring indicator
    y=cbind(time=ty,status=tcens) # y=Surv(ty,1-tcens) with library(survival)

    goldCox = survival::coxph(survival::Surv(y[,1],y[,2]) ~ x)
    cyclopsCoxData = createCyclopsData(survival::Surv(y[,1],y[,2]) ~ x, modelType = "cox")
    cyclopsCox = fitCyclopsModel(cyclopsCoxData)

    x = x[order(y[,1]),]
    y = y[order(y[,1]),]
    di = y[,2]
    i = if(intercept) 1 else 0

    goldIterCounts = c()
    goldIterBeta = rep(0, dim(x)[2] + i)
    goldIterPoisson = 0
    goldBetaLast = rep(0, dim(x)[2] + i)
    goldBetaNext = rep(0, dim(x)[2] + i)

    cyclopsIterCounts = c()
    cyclopsIterBeta = rep(0, dim(x)[2] + i)
    cyclopsIterPoisson = 0
    cyclopsBetaLast = rep(0, dim(x)[2] + i)
    cyclopsBetaNext = rep(0, dim(x)[2] + i)

    while((dist(rbind(goldBetaLast, goldBetaNext)) > e) | (goldIterPoisson == 0)) {
        goldIterPoisson = goldIterPoisson + 1
        goldBetaLast = goldBetaNext
        tj = exp(x %*% goldBetaLast[(1+i):(p+i)])
        si = sapply(1:N, FUN = function(k){return(sum(tj[k:N]))})
        ki = sapply(1:N, FUN = function(k){return(sum(di[1:k] / si[1:k]))})

        counts = di/ki
        if (!all(di[ki==0]==0)) stop("ki 0 but di not 0")
        counts[ki==0] = 0

        if (i == 0) {
            fitPoisson = glm(counts ~ x - 1, family = poisson(), weights = ki, start = goldBetaLast)
        } else {
            fitPoisson = glm(counts ~ x, family = poisson(), weights = ki, start = goldBetaLast)
        }

        goldBetaNext = fitPoisson$coefficients
        goldIterBeta = cbind(goldIterBeta, goldBetaNext)

        goldIterCounts = c(goldIterCounts, fitPoisson$iter)
    }

    while((dist(rbind(cyclopsBetaLast, cyclopsBetaNext)) > e) | (cyclopsIterPoisson == 0)) {
        cyclopsIterPoisson = cyclopsIterPoisson + 1
        cyclopsBetaLast = cyclopsBetaNext
        tj = exp(x %*% cyclopsBetaLast[(1+i):(p+i)])
        si = sapply(1:N, FUN = function(k){return(sum(tj[k:N]))})
        ki = sapply(1:N, FUN = function(k){return(sum(di[1:k] / si[1:k]))})

        counts = di/ki
        if (!all(di[ki==0]==0)) stop("ki 0 but di not 0")
        counts[ki==0] = 0

        if (i == 0) {
            data = createCyclopsData(counts ~ x - 1, modelType = "pr")
        } else {
            data = createCyclopsData(counts ~ x, modelType = "pr")
        }
        fitPoisson = fitCyclopsModel(data, startingCoefficients = cyclopsBetaLast, weights = ki,
                                     prior = createPrior("none"), control = createControl(noiseLevel = "silent"))

        cyclopsBetaNext = coef(fitPoisson)
        cyclopsIterBeta = cbind(cyclopsIterBeta, cyclopsBetaNext)

        cyclopsIterCounts = c(cyclopsIterCounts, fitPoisson$iterations)
    }

    colnames(goldIterBeta) = 0:goldIterPoisson
    colnames(cyclopsIterBeta) = 0:cyclopsIterPoisson

    goldPoisson = list(coefficients = goldBetaNext, iterations = sum(goldIterCounts),
                       runs = goldIterPoisson, iterCounts = goldIterCounts, beta = goldIterBeta)

    cyclopsPoisson = list(coefficients = cyclopsBetaNext, iterations = sum(cyclopsIterCounts),
                       runs = cyclopsIterPoisson, iterCounts = cyclopsIterCounts, beta = cyclopsIterBeta)

    result = list(seedCoefficients = beta,
                  goldCox = goldCox,
                  cyclopsCox = cyclopsCox,
                  goldPoisson = goldPoisson,
                  cyclopsPoisson = cyclopsPoisson)

    class(result) = "coxMM"
    return(result)
}

#' @export
coef.coxMM <- function(x) {
    result = list(seed = x$seedCoefficients,
                  goldCox = x$goldCox$coefficients,
                  cyclopsCox = coef(x$cyclopsCox),
                  goldPoisson = x$goldPoisson$coefficients,
                  cyclopsPoisson = x$cyclopsPoisson$coefficients)
    if (names(x$goldPoisson$coefficients)[1] == "(Intercept)") {
        result$goldPoisson = result$goldPoisson[-1]
        result$cyclopsPoisson = result$cyclopsPoisson[-1]
    }
    return(result)
}

#' @export
iterCoxMM <- function(x) {
    result = list(goldCox = x$goldCox$iter,
                  cyclopsCox = x$cyclopsCox$iterations,
                  goldPoisson = x$goldPoisson$iterations,
                  cyclopsPoisson = x$cyclopsPoisson$iterations)
    return(result)
}

#' @export
distCoxMM <- function(x) {
    return(dist(rbind(coef(x)$cyclopsCox, coef(x)$cyclopsPoisson))[[1]])
}

