## Resubmission
This is a resubmission. In this version I have:

* Fixed a build ERROR under `solaris`.  The error was:
    cyclops/engine/Recursions.hpp:113:24: error: call of overloaded ‘pow(int, const int&)’ is ambiguous
  Solution: ints are now explicitly cast to doubles.

* Fixed a build ERROR under `solaris`.  The error was:
    cyclops/ModelData.cpp:492:22: error: expected unqualified-id before numeric constant auto SS = column.accumulate(squaredSumOp, 0.0);
  Solution: SS seems to be a pre-defined macro; changed variable name.

## Test environments
* local OS X install, R 3.2.2
* ubuntu 12.04 (on travis-ci), R 3.2.5
* win-builder (devel and release)

## R CMD check results
There were no ERRORs or WARNINGs.

There was 1 NOTE:

* checking CRAN incoming feasibility ... NOTE
  Maintainer: 'Marc A. Suchard <msuchard@ucla.edu>'

  New submission

  Possibly mis-spelled words in DESCRIPTION:
    datasets (14:36)
    healthcare (12:40)
    majorization (11:5)
    parallelization (13:50)
    
- These words are spelled correctly, albeit the latter two in American English 

## Downstream dependencies
There are currently no downstream dependencies.
