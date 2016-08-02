## ASAP patch requested by Ripley for `solaris`
This is a patch of 1.2.0 (from yesterday). In this version I have:

* Fixed a build ERROR under `solaris`.  The error was:
    cyclops/engine/Recursions.hpp:113:24: error: call of overloaded ‘pow(int, const int&)’ is ambiguous
  - Solution: ints are now explicitly cast to doubles.

* Fixed a build ERROR under `solaris`.  The error was:
    cyclops/ModelData.cpp:492:22: error: expected unqualified-id before numeric constant auto SS = column.accumulate(squaredSumOp, 0.0);
  - Solution: SS is a pre-defined macro; changed variable name.

* Attempted to decrease compiled library size on `linux`.  
  On these systems, `-g` is default and generates large binaries.
  - Solution: add `-g0` to `Makevars` (does not appear to work)

## Test environments
* local OS X install, R 3.2.2
* ubuntu 12.04 (on travis-ci), R 3.2.5
* win-builder (devel and release)

## R CMD check results
There were no ERRORs or WARNINGs.

There were 2 NOTEs (on `linux` systems, only 1 on `win` and `mac`):

* checking CRAN incoming feasibility ... NOTE
Maintainer: 'Marc A. Suchard <msuchard@ucla.edu>'

Days since last update: 1

 - Brian Ripley emailed me to ask for a `solaris` build fix ASAP

Possibly mis-spelled words in DESCRIPTION:
  datasets (14:36)
  healthcare (12:40)
  majorization (11:5)
  parallelization (13:50)
  
  - These words are spelled correctly, albeit the latter two in American English 

* checking installed package size ... NOTE
  installed size is 20.4Mb
  sub-directories of 1Mb or more:
    libs  19.6Mb
    
  - Package code is highly templated C++ to generate computationally efficient code for five regression models using dense, sparse and binary covariate matrices.  Including debug information via `-g` generates large object files.  Disabling debug information via `-g0` does not appear to work on travis-ci.  Mac and Windows `R` builds do not use `-g`.

## Downstream dependencies
There are currently no downstream dependencies.
