#' @method survfit cyclopsFit
#' @title Calculate baseline hazard function
#'
#' @description
#' \code{survfit.cyclopsFit} computes baseline hazard function
#'
#' @details
#' For \code{cox} and \code{cox_counting} fits, the Breslow estimate uses the case weights of the fit in
#' both the event counts and the risk sets, and gives one curve per stratum, as
#' \code{survfit(coxph(..., weights = , ties = "breslow"), ctype = 1, stype = 2)}. Rows with weight 0
#' are left out. The curves are for the mean covariates of all rows, without the weights (as
#' \code{survfit.coxph} for counting-process data; for right-censored data with weights it uses the
#' weighted means). \code{\link{predict.cyclopsFit}} gives the survival of given covariates. It stops with an
#' error once another model has been fitted to the same Cyclops data, unless that fit used
#' \code{forceNewObject = TRUE}.
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

    if (type == "aalen" && cyclopsFit$cyclopsData$modelType %in% c("cox", "cox_counting")) {
        # One curve per stratum, concatenated as in survfit.coxph
        curves <- .breslowCurves(cyclopsFit, meanLinearPredictor(cyclopsFit))
        result <- list(time = unlist(lapply(curves, `[[`, "time"), use.names = FALSE),
                       surv = exp(-unlist(lapply(curves, `[[`, "cumhaz"), use.names = FALSE)))
        if (length(curves) > 1) {
            result$strata <- stats::setNames(sapply(curves, function(curve) length(curve$time)),
                                             paste0("stratum=", names(curves)))
        }
        return(result)
    }

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

# Weighted Breslow cumulative hazards of a cox or cox_counting fit at linear predictor delta, one curve
# (time, cumhaz) per stratum, named by the internal stratum index, at the stop times of the rows with
# positive weight
.breslowCurves <- function(cyclopsFit, delta, linearPredictor = .fittedLinearPredictor(cyclopsFit)) {
    cyclopsData <- cyclopsFit$cyclopsData
    times <- getTimeVector(cyclopsData)
    events <- getYVector(cyclopsData)
    startTimes <- if (cyclopsData$modelType == "cox_counting") getStartTimeVector(cyclopsData) else numeric(0)
    if (length(startTimes) == 0) {
        startTimes <- rep(-Inf, length(times)) # at risk from the time origin
    }
    strata <- .cyclopsGetStratumVector(cyclopsData)
    if (length(strata) == 0) {
        strata <- rep(0L, length(times))
    }
    weights <- cyclopsFit$weights # data order; NULL when every row has weight 1
    if (is.null(weights)) {
        weights <- rep(1, length(times))
    }
    predictors <- exp(linearPredictor - delta)

    inFit <- which(weights > 0)
    rows <- split(inFit, factor(strata[inFit], levels = unique(strata[inFit])))
    lapply(rows, function(k) {
        .breslowCumulativeHazard(startTimes[k], times[k], events[k], weights[k], predictors[k])
    })
}

# The linear predictors of the fitted rows (data order) of a fit, from its interface. Every fit on a cyclopsData
# shares one interface, so stop when a later fit there has replaced the coefficients or weights of this one
.fittedLinearPredictor <- function(cyclopsFit) {
    interface <- cyclopsFit$interface
    if (.isRcppPtrNull(interface)) {
        stop("Interface object is not initialized")
    }
    beta <- as.numeric(coef(cyclopsFit))
    first <- if (.cyclopsGetHasOffset(cyclopsFit$cyclopsData)) 1 else 0 # index 0 is then the offset
    current <- vapply(seq_along(beta) - 1 + first, function(i) .cyclopsGetBeta(interface, i), numeric(1))
    if (!identical(current, beta) ||
        !identical(.cyclopsGetWeights(interface), as.numeric(cyclopsFit$weights))) { # empty when unweighted
        stop("Another model was fitted to the data of this fit after it; refit this model, ",
             "or fit later models with forceNewObject = TRUE")
    }
    unname(.cyclopsPredictModel(interface)$prediction)
}

# Breslow cumulative hazard over (start, stop] risk sets with case weights w, at the distinct stop times:
# the weighted events at s over D(s), where D(s) = sum of w exp(x'beta) over start < s, minus the sum over stop < s
.breslowCumulativeHazard <- function(start, stop, events, weights, predictors) {
    risk <- weights * predictors
    isEvent <- events == 1
    eventTimes <- sort(unique(stop[isEvent]))
    byStart <- order(start)
    byStop <- order(stop)
    denominator <-
        c(0, cumsum(risk[byStart]))[findInterval(eventTimes, start[byStart], left.open = TRUE) + 1] -
        c(0, cumsum(risk[byStop]))[findInterval(eventTimes, stop[byStop], left.open = TRUE) + 1]
    nEvents <- as.vector(rowsum(weights[isEvent], match(stop[isEvent], eventTimes)))

    newTimes <- sort(unique(stop))
    list(time = newTimes,
         cumhaz = c(0, cumsum(nEvents / denominator))[findInterval(newTimes, eventTimes) + 1])
}
