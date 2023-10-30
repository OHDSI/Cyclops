#' @title Split the analysis time into several intervals for time-varying coefficients.
#'
#' @description
#' \code{splitTime} split the analysis time into several intervals for time-varying coefficients
#'
#' @param shortOut     A data frame containing the outcomes with predefined columns (see below).
#' @param cut          Numeric: Time points to cut at
#'
#' These columns are expected in the shortOut object:
#' \tabular{lll}{
#'   \verb{rowId}   \tab(integer) \tab Row ID is used to link multiple covariates (x) to a single outcome (y) \cr
#'   \verb{y}       \tab(real) \tab Observed event status \cr
#'   \verb{time}    \tab(real) \tab Observed event time \cr
#' }
#'
#' @return A long outcome table for time-varying coefficients.
#' @importFrom survival survfit survSplit
#' @export
splitTime <- function(shortOut, cut) {

    if (!"time" %in% colnames(shortOut)) stop("Must provide observed event time.")
    if (!"y" %in% colnames(shortOut)) stop("Must provide observed event status.")
    if ("rowId" %in% colnames(shortOut)) {
        shortOut <- shortOut %>%
            rename(subjectId = .data$rowId) %>%
            arrange(.data$subjectId)
    } else {
        shortOut <- shortOut %>%
            mutate(subjectId = row_number())
    }

    shortOut <- collect(shortOut)
    longOut <- do.call('survSplit', list(formula = Surv(shortOut$time, shortOut$y)~.,
                                         data = shortOut,
                                         cut = cut,
                                         episode = "stratumId",
                                         id = "newSubjectId"))
    longOut <- longOut %>%
        rename(y = .data$event) %>%
        mutate(time = .data$tstop - .data$tstart) %>%
        select(-c(.data$newSubjectId, .data$tstart, .data$tstop)) %>%
        arrange(.data$stratumId, .data$subjectId)

    # Restore rowIds
    SubjectIds <- shortOut$subjectId
    newSubjectId <- max(SubjectIds)+1
    longOut$rowId <-c(SubjectIds, # rowId = subjectId at 1st stratum
                      newSubjectId:(newSubjectId+(nrow(longOut)-length(SubjectIds))-1)) # create new distinct rowIds for other strata

    # Reorder columns
    longOut <- longOut %>%
        select(.data$stratumId, .data$subjectId, .data$rowId, everything())
        # select(.data$rowId, everything()) %>%
        # select(.data$subjectId, everything()) %>%
        # select(.data$stratumId, everything())

    return(longOut)
}

#' @title Convert short sparse covariate table to long sparse covariate table for time-varying coefficients.
#'
#' @description
#' \code{convertToTimeVaryingCoef} convert short sparse covariate table to long sparse covariate table for time-varying coefficients.
#'
#' @param shortCov       A data frame containing the covariate with predefined columns (see below).
#' @param longOut        A data frame containing the outcomes with predefined columns (see below), output of \code{splitTime}.
#' @param timeVaryCoefId   Integer: A numeric identifier of a time-varying coefficient
#'
#' @details
#' These columns are expected in the shortCov object:
#' \tabular{lll}{
#'   \verb{rowId}  	       \tab(integer) \tab Row ID is used to link multiple covariates (x) to a single outcome (y) \cr
#'   \verb{covariateId}    \tab(integer) \tab A numeric identifier of a covariate  \cr
#'   \verb{covariateValue} \tab(real) \tab The value of the specified covariate \cr
#' }
#'
#' These columns are expected in the longOut object:
#' \tabular{lll}{
#'   \verb{stratumId}   \tab(integer) \tab Stratum ID for time-varying models \cr
#'   \verb{subjectId}  	\tab(integer) \tab Subject ID is used to link multiple covariates (x) at different time intervals to a single subject \cr
#'   \verb{rowId}  	    \tab(integer) \tab Row ID is used to link multiple covariates (x) to a single outcome (y) \cr
#'   \verb{y}           \tab(real) \tab The outcome variable \cr
#'   \verb{time}        \tab(real) \tab For models that use time (e.g. Poisson or Cox regression) this contains time \cr
#'                      \tab       \tab(e.g. number of days) \cr
#' }
#' @return A long sparse covariate table for time-varying coefficients.
#' @export
convertToTimeVaryingCoef <- function(shortCov, longOut, timeVaryCoefId) {

    # Process time-varying coefficients
    timeVaryCoefId <- sort(unique(timeVaryCoefId))
    numTime <- length(timeVaryCoefId) # number of time-varying covariates
    maxCovId <- max(shortCov$covariateId)

    # First stratum
    longCov <- shortCov
    longCov$stratumId <- 1
    colnames(longCov)[which(names(longCov) == "rowId")] <- "subjectId"
    colnames(shortCov)[which(names(shortCov) == "rowId")] <- "subjectId"

    # Rest of strata
    maxStrata <- max(longOut$stratumId)
    for (st in 2:maxStrata) {

        # get valid subjects in current stratum
        subId <- longOut[longOut$stratumId == st, ]$subjectId

        # get valid sparse covariates information in current stratum
        curStrata <- shortCov[shortCov$subjectId %in% subId, ]

        if (any(curStrata$covariateId %in% timeVaryCoefId)) { # skip when valid subjects only have non-zero time-indep covariates
            curStrata$stratumId <- st # assign current stratumId

            # recode covariateId for time-varying coefficients
            # TODO update label
            for (i in 1:numTime) {
                curStrata[curStrata$covariateId == timeVaryCoefId[i], "covariateId"] <- maxCovId + numTime * (st - 2) + i
            }

            # bind current stratum to longCov
            longCov <- rbind(longCov, curStrata)
        }
    }

    # match rowId in longCov
    longCov$rowId <- NA
    for (i in 1:nrow(longCov)) {
        longCov$rowId[i] <- longOut[with(longOut, subjectId == longCov$subjectId[i] & stratumId == longCov$stratumId[i]), "rowId"]
    }

    return(longCov)
}

