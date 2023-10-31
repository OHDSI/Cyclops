#' @title Set GPU device
#'
#' @description
#' \code{setOpenCLDevice} set GPU device
#'
#' @param name String: Name of GPU device
#'
#' @export
setOpenCLDevice <- function(name) {
    devices <- listOpenCLDevices()

    if (!(name %in% devices)) {
        stop("Unable to find device.")
    }

    Sys.setenv(BOOST_COMPUTE_DEFAULT_DEVICE = name)
}
