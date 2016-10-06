## Resubmission
This is a resubmission of version 1.2.1 (original submission in August). In this version I have:

* Fixed a build ERROR under c++14 via g++-6.  The error was:
    cyclops/CompressedDataMatrix.h:406:34: error: call of overloaded ‘make_unique(bsccs::IntVectorPtr&, bsccs::RealVectorPtr&, bsccs::FormatType&)’ is ambiguous
  - Solution: 
    1. I have if-guarded my c++11 work-around for `make_unique()`
    2. I have added a travis-ci build that compiles under c++14 to check that there are no standing forward-compatibility issues and to help limit future errors.
    3. I thank Brian Ripley for taking the time to find this issue; I am a big fan of c++14 and hope to see more use of it in R.    
* Fixed multiple ASan and UBSan warnings


## Test environments
* local OS X install, R 3.2.2
* ubuntu 12.04 (on travis-ci), R 3.2.5, gcc 4.6.3 and gcc 6.0
* win-builder (devel and release)

## R CMD check results
There were no ERRORs or WARNINGs.

There were 1 NOTE:

* checking CRAN incoming feasibility ... NOTE
Maintainer: 'Marc A. Suchard <msuchard@ucla.edu>'

Possibly mis-spelled words in DESCRIPTION:
  datasets (14:36)
  healthcare (12:40)
  majorization (11:5)
  parallelization (13:50)
  
  - These words are spelled correctly, albeit the latter two in American English 

## Downstream dependencies
There are currently no downstream dependencies.
