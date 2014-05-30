library(testthat)
library(CCD)

#test_package("CCD")

a = ccd_model_data(c(1:10), c(1:10), c(1:10), c(1:10), Matrix(rep(1,10),nrow=1), NULL, NULL)
b = ccdInitializeModelImpl(a[[1]])
