Cyclops
=======

[![Build Status](https://travis-ci.org/OHDSI/Cyclops.svg?branch=master)](https://travis-ci.org/OHDSI/Cyclops)
[![codecov.io](https://codecov.io/github/OHDSI/Cyclops/coverage.svg?branch=master)](https://codecov.io/github/OHDSI/Cyclops?branch=master)
[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/Cyclops)](https://CRAN.R-project.org/package=Cyclops)
[![CRAN_Status_Badge](http://cranlogs.r-pkg.org/badges/Cyclops)](https://cran.r-project.org/package=Cyclops)

Cyclops is part of the [OHDSI Methods Library](https://ohdsi.github.io/MethodsLibrary).


Introduction
============

Cyclops (Cyclic coordinate descent for logistic, Poisson and survival analysis) is an R package for performing large scale regularized regressions.

Features
========
 - Regression of very large problems: up to millions of observations, millions of variables
 - Supports (conditional) logistic regression, (conditional) Poisson regression, as well as (conditional) Cox regression
 - Uses a sparse representation of the independent variables when appropriate
 - Supports using no prior, a normal prior or a Laplace prior
 - Supports automatic selection of hyperparameter through cross-validation
 - Efficient estimation of confidence intervals for a single variable using a profile-likelihood for that variable

Examples
========

```r
  library(Cyclops)
  cyclopsData <- createCyclopsDataFrame(formula)
  cyclopsFit <- fitCyclopsModel(cyclopsData)
```
 
Technology
============
Cyclops in an R package, with most functionality implemented in C++. Cyclops uses cyclic coordinate descent to optimize the likelihood function, which makes use of the sparse nature of the data.

System Requirements
===================
Requires R (version 3.1.0 or higher). Compilation on Windows requires [RTools >= 3.4]( https://CRAN.R-project.org/bin/windows/Rtools/).

Installation
============
In R, to install the latest stable version, install from CRAN:

```r
install.packages("Cyclops")
```
  
To install the latest development version, install from GitHub. Note that this will require RTools to be installed.

```r
install.packages("devtools")
devtools::install_github("ohdsi/Cyclops")
```

 
User Documentation
==================
* Package manual: [Cyclops manual](https://raw.githubusercontent.com/OHDSI/Cyclops/master/extras/Cyclops.pdf) 

Support
=======
* Developer questions/comments/feedback: <a href="http://forums.ohdsi.org/c/developers">OHDSI Forum</a>
* We use the <a href="https://github.com/OHDSI/Cyclops/issues">GitHub issue tracker</a> for all bugs/issues/enhancements
 
License
=======
Cyclops is licensed under Apache License 2.0.   Cyclops contains the TinyThread libray.

The TinyThread library is licensed under the [zlib/libpng](https://opensource.org/licenses/Zlib) license as described [here](http://tinythreadpp.bitsnbites.eu).


Development
===========
Cyclops is being developed in R Studio.

### Development status

Beta

Acknowledgements
================
- This project is supported in part through the National Science Foundation grants IIS 1251151 and DMS 1264153.


