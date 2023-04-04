## Initial submission of patch update to package

* fixed all CRAN WARNINGS, including generic-inconsistency, uninitialized values

## Test environments
* local OS X install, R 4.1
* ubuntu 20.04 (via gh-actions: devel and release)
* win-builder (devel and release)

## R CMD check results
* There were no ERRORs or WARNINGs
* There is 2 occasional NOTEs:
  1. checking installed package size ... NOTE
    installed size is 22.5Mb
    sub-directories of 1Mb or more:
      libs 21.7Mb
  2. Specified C++11: please drop specification unless essential      

The first NOTE occurs on systems (like 'r-devel-linux-x86_64-fedora-clang') that include debug
symbols in their compilation; Cyclops performance is heavily dependent on many template
instantiations that generate a large library when including debug symbols.  Future
availability of C++17 'if (constexpr ...)' should decrease library size substantially.

The second NOTE remains because many OHDSI community users remain on R 4.0 that supports C++11
but does not automatically compile with C++11 enabled.  

## Downstream dependencies
* 'EvidenceSynthesis' - checked and works.
* 'IterativeHardThresholding' - checked and works.
