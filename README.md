Cyclops
=======

Introduction
============

Cyclops (Cyclic coordinate descent for logistic, Poisson and survival analysis) is an R package for performing large scale regularized regressions.

Features
========
 - Regression of very large problems: millions of observations, millions of variables
 - Supports (conditional) logistic regression, (conditional) Poisson regression, as well as (conditional) Cox regression
 - Uses a sparse representation of the independent variables when appropriate
 - Supports using no prior or a LaPlace prior
 - Supports automatic selection of hyperparameter through cross-validation
 - Efficient estimation of confidence intervals for a single variable by turning of regularization for that variable

Examples
========

```r
  library(Cyclops)
  cyclopsData <- createCyclopsDataFrame(formula)
  cyclopsFit <- fitCyclopsModel(cyclopsData)
```
 
Technology
============
Cyclops in an R package, with most functionality implemented in C++.

System Requirements
===================
Requires R (version 3.1.0 or higher). Installation on Windows requires [RTools](http://cran.r-project.org/bin/windows/Rtools/).

Dependencies
============
 * There are no dependencies.

Getting Started
===============
1. On Windows, make sure [RTools](http://cran.r-project.org/bin/windows/Rtools/) is installed.
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
* Package manual: [Cyclops manual](https://raw.githubusercontent.com/OHDSI/Cyclops/master/man/Cyclops.pdf) 
* Developer questions/comments/feedback: <a href="http://forums.ohdsi.org/c/developers">OHDSI Forum</a>
* We use the <a href="../../issues">GitHub issue tracker</a> for all bugs/issues/enhancements
 
License
=======
Achilles is licensed under Apache License 2.0

Development
===========
Cyclops is being developed in R Studio.

###Development status

Alpha

Acknowledgements
================
- This project is supported in part through the National Science Foundation grants IIS 1251151 and DMS 1264153.
