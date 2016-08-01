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

## Resubmission
This is a resubmission. In this version I have:

* Deleted `Cyclops-Ex.R` at top level.  My apologies here! I was testing with `valgrind` in between `build_win()` and `release()`.
