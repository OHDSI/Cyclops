% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Survfit.R
\name{survfit.cyclopsFit}
\alias{survfit.cyclopsFit}
\title{Calculate baseline hazard function}
\usage{
\method{survfit}{cyclopsFit}(formula, type = "aalen", ...)
}
\arguments{
\item{formula}{A Cyclops survival model fit object}

\item{type}{type of baseline survival, choices are: "aalen" (Breslow)}

\item{...}{for future methods}
}
\value{
Baseline survival function for mean covariates
}
\description{
\code{survfit.cyclopsFit} computes baseline hazard function
}
