% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Multitype.R
\name{Multitype}
\alias{Multitype}
\title{Create a multitype outcome object}
\usage{
Multitype(y, type)
}
\arguments{
\item{y}{Numeric: Response count(s)}

\item{type}{Numeric or factor: Response type}
}
\value{
An object of class \code{Multitype} with length equal to the length of \code{y} and \code{type}.
}
\description{
\code{Multitype} creates a multitype outcome object, usually used as a response variable in a
hierarchical Cyclops model fit.
}
\examples{
Multitype(c(0,1,0), as.factor(c("A","A","B")))

}
