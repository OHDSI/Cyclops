#' @method survfit cyclopsFit
#' @title Calculate baseline hazard function
#'
#' @description
#' \code{survfit.cyclopsFit} computes baseline hazard function
#'
#' @param formula A Cyclops survival model fit object
#'
#' @return Baseline survival function, not adjusted for each covariate to have mean 0 (unlike coxph{survival})
#'
#' @importFrom survival survfit
#'
#' @export
survfit.cyclopsFit <- function(formula) {
    times = getTimeVector(formula$cyclopsData)
    events = getYVector(formula$cyclopsData)
    predictors = exp(predict(formula))

    accDenom = Reduce("+",predictors,accumulate=TRUE)
    newAccDenom = accDenom[length(times)+1-match(unique(times),times[length(times):1])]
    newTimes = unique(times)
    n = length(newTimes)

    t = table(times*events)[-1]
    g = match(as.numeric(names(t)), newTimes)
    newY = rep(0,n)
    newY[g] = t

    l = newY/newAccDenom
    L = rep(0,n)
    for (i in 1:n) {
        L[n+1-i] = sum(l[i:n])
    }
    return (list(time = newTimes[order(newTimes)],
                 surv = 1/exp(L)))
}
