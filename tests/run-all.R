library(testthat)
library(CCD)

#test_package("CCD")

a = ccd_model_data(NULL, c(1:10), NULL, NULL, Matrix(rep(1,10),nrow=1), NULL, NULL)
b = ccdInitializeModelImpl(a[[1]])
