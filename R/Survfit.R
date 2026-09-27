#' @method survfit cyclopsFit
#' @title Calculate baseline hazard function
#'
#' @description
#' \code{survfit.cyclopsFit} computes baseline hazard function
#'
#' @param formula A Cyclops survival model fit object
#' @param type type of baseline survival, choices are: "aalen" (Breslow)
#' @param ... for future methods
#'
#' @return Baseline survival function for mean covariates
#'
#' @importFrom survival survfit
#'
#' @export
survfit.cyclopsFit <- function(formula, type="aalen", ...) {
    cyclopsFit <- formula
    delta = meanLinearPredictor(cyclopsFit)

    times = getTimeVector(cyclopsFit$cyclopsData)
    events = getYVector(cyclopsFit$cyclopsData)
    predictors = exp(predict(cyclopsFit))
    startTimes = getStartTimeVector(cyclopsFit$cyclopsData)

    if (type == "aalen" && length(startTimes) > 0) {
        # Breslow increments over (start, stop] risk sets:
        # D(s) = sum over start < s minus sum over stop < s
        u <- unname(predictors)
        eventTimes <- sort(unique(times[events == 1]))
        byStart <- order(startTimes)
        byStop <- order(times)
        denominator <-
            c(0, cumsum(u[byStart]))[findInterval(eventTimes, startTimes[byStart], left.open = TRUE) + 1] -
            c(0, cumsum(u[byStop]))[findInterval(eventTimes, times[byStop], left.open = TRUE) + 1]
        nEvents <- tabulate(match(times[events == 1], eventTimes), length(eventTimes))

        newTimes <- sort(unique(times))
        L <- c(0, cumsum(nEvents / denominator))[findInterval(newTimes, eventTimes) + 1]
        return (list(time = newTimes,
                     surv = 1/exp(L)^exp(delta)))
    }

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
