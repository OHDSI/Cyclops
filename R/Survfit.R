#' @method survfit cyclopsFit
#' @title Calculate baseline hazard function
#'
#' @description
#' \code{survfit.cyclopsFit} computes baseline hazard function
#'
#' @param cyclopsFit A Cyclops survival model fit object
#' @param type type of baseline survival, choices are: "aalen" (Breslow)
#'
#' @return Baseline survival function for mean covariates
#'
#' @importFrom survival survfit
#'
#' @export
survfit.cyclopsFit <- function(cyclopsFit, type="aalen") {
    delta = meanLinearPredictor(cyclopsFit)

    times = getTimeVector(cyclopsFit$cyclopsData)
    events = getYVector(cyclopsFit$cyclopsData)
    predictors = exp(predict(cyclopsFit))

    if (type == "aalen") {
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
                     surv = 1/exp(L)^exp(delta)))
    }
}
