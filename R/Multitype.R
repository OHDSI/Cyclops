
#' @title Create a multitype outcome object
#' 
#' @description
#' \code{Multitype} creates a multitype outcome object, usually used as a response variable in a 
#' hierarchical Cyclops model fit.
#' 
#' @param y     Numeric: Response count(s)
#' @param type Numeric or factor: Response type
#' 
#' @return An object of class \code{Multitype} with length equal to the length of \code{y} and \code{type}.
#' @examples
#' Multitype(c(0,1,0), as.factor(c("A","A","B")))
Multitype <- function(y, type) {
    if (missing(y) || !is.numeric(y)) stop("Must have outcome counts")
    if (missing(type)) stop("Must have outcome types")
    if (length(y) != length(type)) stop("Must have equal lengths")
    
    #mt <- data.frame(y, type)
    
    #mt$dim <- c(length(y),2)
    #attr(mt, "dim")  <- c(length(y), 2)
    mt <- cbind(y = y, type = type)
    attr(mt, "contrasts") <- contrasts(type)
    class(mt) <- c("Multitype")
    
    mt
}
