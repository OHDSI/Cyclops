#
#

#' @export
setOpenCLDevice <- function(name) {
    devices <- listOpenCLDevices()

    if (!(name %in% devices)) {
        stop("Unable to find device.")
    }

    Sys.setenv(BOOST_COMPUTE_DEFAULT_DEVICE = name)
}
