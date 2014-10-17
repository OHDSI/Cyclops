

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


# "[.Multitype" <- function(x, i, j, drop=FALSE) {
#             class(x) <- 'data.frame'
#         NextMethod("[")
# }

# "[.Multitype" <- function(x, i, j, drop=FALSE) {
#     # If only 1 subscript is given, the result will still be a Surv object,
#     #   and the drop argument is ignored.
#     # I would argue that x[3:4,,drop=FALSE] should return a matrix, since
#     #  the user has implicitly specified that they want a matrix.
#     #  However, [.dataframe calls [.Surv with the extra comma; its
#     #  behavior drives the choice of default.
#     if (missing(j)) {
#         xattr <- attributes(x)
#         x <- unclass(x)[i,, drop=FALSE] # treat it as a matrix: handles dimnames
#         attr(x, 'type') <- xattr$type
#         if (!is.null(xattr$states)) attr(x, "states") <- xattr$states
#         if (!is.null(xattr$inputAttributes)) {
#             # If I see "names" subscript it, leave all else alone
#             attr(x, 'inputAttributes') <- 
#                 lapply(xattr$inputAttributes, function(z) {
#                     if (any(names(z)=="names")) z$names <- z$names[i]
#                     z
#                 })
#         }
#         class(x) <- "Multitype"  #restore the class
#         x
#     }
#     else { # return  a matrix or vector
#         if (is.R()) class(x) <- 'matrix'
#         else oldClass(x) <- NULL
#         NextMethod("[")
#     }
# }