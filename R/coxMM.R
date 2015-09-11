#' Simulates data to test cox MM
#'
#' @description
#' Compares gold standard and cyclops calculations of original cox regression and poisson MM regression
#'
#' @param x covariates
#' @param y outcomes
#' @param e distance between Poisson regressions to stop
#' @param intercept whether to include intercept in model, default = TRUE
#'
#' @return
#' Returns a list with the following: \describe{
#' \item{coefficients}{calculated coefficients for each method}
#' \item{cyclopsCox}{iterations ran for each method}
#' \item{goldPoisson}{distance between the two cox methods, two poisson methods, and between the two cyclops methods} }
#'
#' @export
compareCoxMM <- function(x, y, e, intercept = TRUE) {
    goldCox = survival::coxph(survival::Surv(y[,1],y[,2]) ~ x, ties = "breslow")
    cyclopsCoxData = createCyclopsData(survival::Surv(y[,1],y[,2]) ~ x, modelType = "cox")
    cyclopsCox = fitCyclopsModel(cyclopsCoxData)

    goldPoisson = computeCoxMM(x, y, e, intercept, method = "glm")
    cyclopsPoisson = computeCoxMM(x, y, e, intercept, method = "cyclops")

    coefficients = list(goldCox = goldCox$coefficients,
                        cyclopsCox = coef(cyclopsCox),
                        goldPoisson = goldPoisson$coefficients,
                        cyclopsPoisson = cyclopsPoisson$coefficients)
    iterations = list(goldCox = goldCox$iter,
                      cyclopsCox = cyclopsCox$iterations,
                      goldPoisson = goldPoisson$iterations,
                      cyclopsPoisson = cyclopsPoisson$iterations)
    distances = list(cox = dist(rbind(coefficients$goldCox, coefficients$cyclopsCox)),
                     pois = if (intercept) dist(rbind(coefficients$goldPoisson[-1], coefficients$cyclopsPoisson[-1]))
                            else dist(rbind(coefficients$goldPoisson, coefficients$cyclopsPoisson)),
                     coxVpois = if (intercept) dist(rbind(coefficients$cyclopsCox, coefficients$cyclopsPoisson[-1]))
                            else dist(rbind(coefficients$cyclopsCox, coefficients$cyclopsPoisson)))
    return(list(coefficients = coefficients, iterations = iterations, distances = distances))
}

#' Simulates cox data
#'
#' @description Simulates cox data that can include ties. Represented as a matrix
#'
#' @param N number of patients
#' @param p number of covariates
#' @param c censoring probability
#' @param simulateTies simulate ties by rounding to 3 significant digits
#'
#' @return
#' \describe{\item{covariates}{matrix of covariates}
#' \item{outcomes}{matrix of outcomes including time to event and event type}}
#'
#' @export
simulateCoxData <- function(N, p, c, simulateTies = TRUE) {
    x = matrix(rnorm(N*p), N, p)
    iz=sample(1:(N*p),size=N*p*.85,replace=FALSE)
    x[iz]=0
    beta = rnorm(p)
    fx = x %*% beta
    hx = exp(fx)
    ty = rexp(N,hx)
    if(simulateTies) ty = signif(ty, digits = 3)
    tcens = 1 - rbinom(n = N, prob = c, size = 1)
    y = cbind(time = ty, status = tcens)

    return(list(covariates = x, outcomes = y))
}

#' computes Cox MM
#'
#' @description computes the Cox MM poisson regressions
#'
#' @param x covariate data
#' @param y outcome data
#' @param e stopping criteria - norm of beta
#' @param intercept whether to include intercept
#' @param method "glm" or "cyclops"
#'
#' @return
#' \describe{
#' \item{coefficients}{calculated coefficients}
#' \item{iterations}{total iterations ran}
#' \item{runs}{number of poisson regressions ran}
#' \item{iterCounts}{iterations per poisson regression}}
#' @export
computeCoxMM <- function(x, y, e, intercept, method) {
    N = dim(x)[1]
    p = dim(x)[2]
    x = x[order(y[,1],-y[,2]),]
    y = y[order(y[,1],-y[,2]),]
    ci = y[,2]
    i = if(intercept) 1 else 0

    # find adjustments for ties
    # di = number of values at tie
    # tf = index within tie (minus 1)
    # tb = reverse index within tie (minus 1)
    tf = rep(0, N)
    tf[2:N] = apply(y[1:(N-1),]==y[2:N,], MARGIN = 1, FUN = function(a){if (all(a)) 1 else 0})
    tf = tf * ci
    for (j in 2:N) {
        tf[j] = (tf[j-1] + tf[j]) * tf[j]
    }
    di = tf
    for (j in 1:N) {
        if ((di[j]>0) & (j==N | di[j+1]==0)) {
            di[(j-di[j]):j] = di[(j-di[j]):j] + (di[j]+1):1
        }
    }
    tb = di-tf
    tb[tb==0]=1
    tb = tb-1

    iterCounts = c()
    #iterBeta = rep(0, dim(x)[2] + i)
    iterPoisson = 0
    betaLast = rep(0, dim(x)[2] + i)
    betaNext = rep(0, dim(x)[2] + i)

    while((dist(rbind(betaLast, betaNext)) > e) | (iterPoisson == 0)) {
        iterPoisson = iterPoisson + 1
        betaLast = betaNext
        tj = exp(x %*% betaLast[(1+i):(p+i)])
        si = sapply(1:N, FUN = function(k){return(sum(tj[(k-tf[k]):N]))})
        ki = sapply(1:N, FUN = function(k){return(sum(ci[1:(k+tb[k])] / si[1:(k+tb[k])]))})

        counts = ci/ki
        if (!all(ci[ki==0]==0)) stop("ki 0 but ci not 0")
        counts[ki==0] = 0

        if (method == "glm") {
            if (i == 0) {
                fitPoisson = glm(counts ~ x - 1, family = poisson(), weights = ki, start = betaLast)
            } else {
                fitPoisson = glm(counts ~ x, family = poisson(), weights = ki, start = betaLast)
            }
        }
        if (method == "cyclops") {
            if (i == 0) {
                data = createCyclopsData(counts ~ x - 1, modelType = "pr")
            } else {
                data = createCyclopsData(counts ~ x, modelType = "pr")
            }
            fitPoisson = fitCyclopsModel(data, startingCoefficients = betaLast, weights = ki,
                                         prior = createPrior("none"), control = createControl(noiseLevel = "silent"))
        }

        betaNext = coef(fitPoisson)
        #iterBeta = cbind(iterBeta, betaNext)
        iterCounts = c(iterCounts, fitPoisson$iter)
    }
    #colnames(iterBeta) = 0:iterPoisson

    result = list(coefficients = betaNext,
                  #beta = iterBeta,
                  iterations = sum(iterCounts),
                  runs = iterPoisson,
                  iterCounts = iterCounts)
    return(result)
}
