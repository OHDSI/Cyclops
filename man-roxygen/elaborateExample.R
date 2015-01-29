#' @examples 
#' #Generate some simulated data:
#' sim <- simulateCyclopsData(nstrata = 1, nrows = 1000, ncovars = 2, eCovarsPerRow = 0.5, 
#'                            model = "poisson")
#' cyclopsData <- convertToCyclopsData(sim$outcomes, sim$covariates, modelType = "pr", 
#'                                     addIntercept = TRUE)
#' 
#' #Define the prior and control objects to use cross-validation for finding the 
#' #optimal hyperparameter:
#' prior <- createPrior("laplace", exclude = 0, useCrossValidation = TRUE)
#' control <- createControl(cvType = "auto", noiseLevel = "quiet")
#' 
#' #Fit the model
#' fit <- fitCyclopsModel(cyclopsData,prior = prior, control = control)  
#' 
#' #Find out what the optimal hyperparameter was:
#' getHyperParameter(fit)
#' 
#' #Extract the current log-likelihood, and coefficients
#' logLik(fit)
#' coef(fit)
#' 
#' #We can only retrieve the confidence interval for unregularized coefficients:
#' confint(fit, c(0))
