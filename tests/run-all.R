library(testthat)
library(CCD)

#test_package("CCD")

a = ccdModelData(NULL, c(1:10), NULL, NULL, Matrix(rep(1,10),nrow=1), NULL, NULL)
b = ccdInitializeModel(a[[1]], modelType="ls", computeMLE=TRUE)
c = ccdFitModel(b[[1]])
d = ccdLogModel(b[[1]])


# (m <- Matrix(c(0,0,2:0), 3,5, dimnames=list(LETTERS[1:3],NULL)))
# ## ``extract the nonzero-pattern of (m) into an nMatrix'':
# nm <- as(m, "nsparseMatrix") ## -> will be a "ngCMatrix"
# str(nm) # no 'x' slot

#R -d gdb -e "source('tmp.r')"
