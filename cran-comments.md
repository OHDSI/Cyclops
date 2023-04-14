## Resubmission after VALGRIND feedback from Brian Ripley

* fixed VALGRIND uninitialized value (UB) issue; variable 'priorType' was uninitialized,
  specifically in:

    ==2293460== Conditional jump or move depends on uninitialised value(s)
    ==2293460==    at 0x22CE4848: bsccs::CyclicCoordinateDescent::computeAsymptoticPrecisionMatrix() (packages/tests-vg/Cyclops/src/cyclops/CyclicCoordinateDescent.cpp:1335)
    
* fixed VALGRIND memory leak issue; was caused by calls to '::Rf_error()' instead of 
  'Rcpp::stop()' when handling some error edge-cases

## Test environments
* local OS X install, R 4.1
* r-devel-valgrind docker container
* ubuntu 20.04 (via gh-actions: devel and release)
* win-builder (devel and release)

## R CMD check results
* There were no ERRORs or WARNINGs
* There is 1 occasional NOTE (besides days since last submission):
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
* 'IterativeHardThresholding' - checked and works.
* 'BrokenAdaptiveRidge' - checked and works.
