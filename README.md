Cyclops
=======

[![CRAN_Status_Badge](http://www.r-pkg.org/badges/version/Cyclops)](https://CRAN.R-project.org/package=Cyclops)

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
Requires R (version 3.1.0 or higher). Installation on Windows requires [RTools]( https://CRAN.R-project.org/bin/windows/Rtools/) (`devtools >= 1.12` required for RTools34, otherwise RTools33 works fine).

Dependencies
============
 * There are no dependencies.

Getting Started
===============
1. On Windows, make sure [RTools](https://CRAN.R-project.org/bin/windows/Rtools/) is installed.
2. In R, use the following commands to download and install Cyclops:

  ```r
  install.packages("devtools")
  library(devtools)
  install_github("ohdsi/Cyclops") 
  ```

3. To perform a Cyclops model fit, use the following commands in R:

  ```r
  library(Cyclops)
  cyclopsData <- createCyclopsDataFrame(formula)
  cyclopsFit <- fitCyclopsModel(cyclopsData)
  ```
 
Getting Involved
================
* Package manual: [Cyclops manual](https://raw.githubusercontent.com/OHDSI/Cyclops/master/extras/Cyclops.pdf) 
* Developer questions/comments/feedback: <a href="http://forums.ohdsi.org/c/developers">OHDSI Forum</a>
* We use the <a href="../../issues">GitHub issue tracker</a> for all bugs/issues/enhancements
 
License
=======
Cyclops is licensed under Apache License 2.0.   Cyclops contains the TinyThread libray.

The TinyThread library is licensed under the [zlib/libpng](https://opensource.org/licenses/Zlib) license as described [here](http://tinythreadpp.bitsnbites.eu).


Development
===========
Cyclops is being developed in R Studio.

### Development status

[![Build Status](https://travis-ci.org/OHDSI/Cyclops.svg?branch=master)](https://travis-ci.org/OHDSI/Cyclops)
[![codecov.io](https://codecov.io/github/OHDSI/Cyclops/coverage.svg?branch=master)](https://codecov.io/github/OHDSI/Cyclops?branch=master)

Beta

Acknowledgements
================
- This project is supported in part through the National Science Foundation grants IIS 1251151 and DMS 1264153.


