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

    if (type == "aalen" && cyclopsFit$cyclopsData$modelType == "cox_counting") {
        # One curve per stratum, concatenated as in survfit.coxph
        startTimes <- getStartTimeVector(cyclopsFit$cyclopsData)
        if (length(startTimes) == 0) {
            startTimes <- rep(-Inf, length(times)) # at risk from the time origin
        }
        strata <- .cyclopsGetStratumVector(cyclopsFit$cyclopsData)
        rows <- split(seq_along(times), factor(strata, levels = unique(strata)))
        curves <- lapply(rows, function(k) {
            .breslowSurvival(startTimes[k], times[k], events[k], unname(predictors)[k], delta)
        })
        result <- list(time = unlist(lapply(curves, `[[`, "time"), use.names = FALSE),
                       surv = unlist(lapply(curves, `[[`, "surv"), use.names = FALSE))
        if (length(curves) > 1) {
            result$strata <- stats::setNames(sapply(curves, function(curve) length(curve$time)),
                                             paste0("stratum=", names(curves)))
        }
        return(result)
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

# Breslow baseline survival at linear predictor delta over (start, stop] risk sets:
# D(s) = sum of exp(x'beta) over start < s, minus the sum over stop < s
.breslowSurvival <- function(start, stop, events, predictors, delta) {
    eventTimes <- sort(unique(stop[events == 1]))
    byStart <- order(start)
    byStop <- order(stop)
    denominator <-
        c(0, cumsum(predictors[byStart]))[findInterval(eventTimes, start[byStart], left.open = TRUE) + 1] -
        c(0, cumsum(predictors[byStop]))[findInterval(eventTimes, stop[byStop], left.open = TRUE) + 1]
    nEvents <- tabulate(match(stop[events == 1], eventTimes), length(eventTimes))

    newTimes <- sort(unique(stop))
    L <- c(0, cumsum(nEvents / denominator))[findInterval(newTimes, eventTimes) + 1]
    list(time = newTimes, surv = 1/exp(L)^exp(delta))
}
