#' @return
#'
#' Returns an \code{\link{environment}} of class inheriting from \code{"ccdData"} that
#' contains at least the following objects:
#' \tabular{ll}{
#'      \code{  ccdDataPtr} \tab    An Rcpp \code{externalptr} to the C++ object that holds the data.
#'                                  The object contents does not get written to \code{.Rdata}, so
#'                                  \code{ccdData} are not reusable across restarts of \code{R}. \cr
#'      \code{  ccdInterfacePtr} \tab   An Rcpp \code{externalptr} to the C++ object that fits the data.
#'                                      Again, this is not reusable across \code{R} invokations. \cr
#'      \code{  timeLoad} \tab Amount of time (in seconds) taken to load the data. \cr
#'  }
#'
#' The generic functions \code{\link{print}} and \code{\link{summary}} can be used to 
#' display various summaries of the data.
#'
