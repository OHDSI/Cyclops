## Submission of v3.4.1

* fix some NaN/-Inf inconsistencies with likelihood profiling
    
## Test environments
* local OS X install, R 4.3
* r-devel-valgrind docker container
* ubuntu 20.04 (via gh-actions: devel and release)
* win-builder (devel and release)

## R CMD check results
* There were no ERRORs or WARNINGs   
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
* 'IterativeHardThresholding' - checked and works.
* 'BrokenAdaptiveRidge' - checked and works.
