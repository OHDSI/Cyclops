## New submission following email from Kurt Hornik reporting unit-test errors likely
originating from `stringsAsFactors = FALSE` by default in upcoming `R 4.0.0`.

* all unit-tests now pass using `R Under development (unstable) (2020-03-01 r77880)`
* several small bug fixes when fitting conditional Poisson models

## Test environments
* local OS X install, R 3.6.1
* docker container based on `rocker/r-devel` but compiled with `r77880`
* ubuntu 14.04 (on travis-ci), R 3.5.1, gcc 4.8.4 and gcc 6.0
* win-builder (devel and release)

## R CMD check results
* There were no ERRORs or WARNINGs
* This is 1 occasional NOTE:
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
