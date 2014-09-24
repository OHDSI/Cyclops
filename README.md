Cyclops
========================

Cyclic coordinate descent for logistic, Poisson and survival analysis 

Getting Started
===============

1. In R, use the following commands to install Cyclops:

  ```r
  install.packages("devtools")
  library(devtools)
  install_github("ohdsi/Cyclops") 
  ```

2. To perform a Cyclops model fit, use the following commands in R:

  ```r
  library(Cyclops)
  cyclopsData <- createCyclopsDataFrame(formula)
  cyclopsFit <- fitCyclopsModel(cyclopsData)
  ```
    
License
=======
Achilles is licensed under Apache License 2.0

# Acknowledgements
- This project is supported in part through the National Science Foundation grants IIS 1251151 and DMS 1264153.
