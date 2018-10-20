## Resubmission (of 2.0.0) following email from Brian Ripley asking:
  See https://cran.r-project.org/web/checks/check_results_Cyclops.html . The container-overflow error is new this version.
  Please correct ASAP once all the results are in, and before Oct 3 to safely retain the package on CRAN.

* Both leaks were patched and confirmed using `rocker/r-devel-ubsan-clang` with ASAN enabled.
* Package no longer imports `RcppParallel` as it gives compilation warnings under `gcc8`

## Test environments
* local OS X install, R 3.5.1
* ubuntu 14.04 (on travis-ci), R 3.4.4, gcc 4.8.4 and gcc 6.0
* win-builder (devel and release)

## R CMD check results
* There were no ERRORs or WARNINGs
* There is 1 NOTE:
  checking CRAN incoming feasibility ... NOTE
  Days since last update: 4

See resubmission request above.

* This is also 1 occasional NOTE:
  checking installed package size ... NOTE
    installed size is 22.5Mb
    sub-directories of 1Mb or more:
      libs 21.7Mb

This occurs on systems (like `r-devel-linux-x86_64-fedora-clang`) that include debug
symbols in their compilation; Cyclops performance is heavily dependent on many template
instantiations that generate a large library when including debug symbols.  Future
availability of C++17 `if (constexpr ...)` should decrease library size substantially.

## Downstream dependencies
There are currently no downstream dependencies.
