## Re-submission after 'valgrind' check email from Uwe Ligges:

```
Thanks, we see with valgrind it is better, but still leaking. 
```

## Fixes

* all definite leaks on my M1 and linux (R 4.1) valgrind-versions are now gone.
* replaced calls to '::Rf_error()' with 'Rcpp::stop()' to ensure that destructors get 
  called before returning to R.
* removed calls to JVM through Andromeda package (was an issue with the CRAN system JVM).

## Test environments
* local OS X install, R 4.1
* ubuntu 20.04 (via gh-actions: devel and release)
* win-builder (devel and release)

## R CMD check results
* There were no ERRORs
* There is 1 occasional WARNING:
  inclusion of 'abort'.
  
This inclusion comes from 'RcppEigen' and not 'Cyclops' on some platforms with R-devel.  
  
* There is 1 occasional NOTE:
  checking installed package size ... NOTE
    installed size is 22.5Mb
    sub-directories of 1Mb or more:
      libs 21.7Mb

This occurs on systems (like 'r-devel-linux-x86_64-fedora-clang') that include debug
symbols in their compilation; Cyclops performance is heavily dependent on many template
instantiations that generate a large library when including debug symbols.  Future
availability of C++17 'if (constexpr ...)' should decrease library size substantially.

## Downstream dependencies
* 'EvidenceSynthesis' - checked and works.
* 'EmpiricalCalibration' - checked and works.
