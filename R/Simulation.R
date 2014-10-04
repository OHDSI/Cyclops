
simulateData <- function(nstrata = 200, 
                         nrows = 10000, 
                         ncovars = 20,
                         effectSizeSd = 1,
                         eCovarsPerRow = ncovars/100,
                         model="survival"){
    
    effectSizes <- data.frame(covariate_id=1:ncovars,rr=exp(rnorm(ncovars,mean=0,sd=effectSizeSd)))
    
    covarsPerRow <- rpois(nrows,eCovarsPerRow)
    covarsPerRow[covarsPerRow > ncovars] <- ncovars
    covarsPerRow <- data.frame(covarsPerRow = covarsPerRow)
    covarRows <- sum(covarsPerRow$covarsPerRow)
    covariates <- data.frame(row_id = rep(0,covarRows), covariate_id = rep(0,covarRows), covariate_value = rep(1,covarRows))
    cursor <- 1
    for (i in 1:nrow(covarsPerRow)){
        n <- covarsPerRow$covarsPerRow[i]
        if (n != 0){
            covariates$row_id[cursor:(cursor+n-1)] <- i
            covariates$covariate_id[cursor:(cursor+n-1)] <- sample.int(size=n,ncovars)
            cursor = cursor+n
        }
    }
    
    outcomes <- data.frame(row_id = 1:nrows, stratum_id = round(runif(nrows,min=1,max=nstrata)), y=0)
    covariates <- merge(covariates,outcomes[,c("row_id","stratum_id")])
    
    row_id_to_rr <- aggregate(rr ~ row_id, data=merge(covariates,effectSizes),prod)
    
    outcomes <- merge(outcomes,row_id_to_rr,all.x=TRUE)
    outcomes$rr[is.na(outcomes$rr)] <- 1
    
    if (model == "survival"){
        strataBackgroundProb <- runif(nstrata,min=0.01,max=0.03)
        outcomes$rate <-  strataBackgroundProb[outcomes$stratum_id] * outcomes$rr
        outcomes$time_to_outcome <- 1+round(rexp(n=nrow(outcomes),outcomes$rate))
        outcomes$time_to_censor <- 1+round(runif(n=nrow(outcomes),min=0,max=499))
        outcomes$time <- outcomes$time_to_outcome
        outcomes$time[outcomes$time_to_censor < outcomes$time_to_outcome] <- outcomes$time_to_censor[outcomes$time_to_censor < outcomes$time_to_outcome]
        outcomes$y <- as.integer(outcomes$time_to_censor > outcomes$time_to_outcome)
    } else if (model == "logistic") {
        strataBackgroundProb <- runif(nstrata,min=0.1,max=0.3)    
        outcomes$prob <-  strataBackgroundProb[outcomes$stratum_id] * outcomes$rr
        outcomes$y <- as.integer(runif(nrows,min=0,max=1) < outcomes$prob)
    } else if (model == "poisson"){
        strataBackgroundProb <- runif(nstrata,min=0.01,max=0.03)
        outcomes$rate <-  strataBackgroundProb[outcomes$stratum_id] * outcomes$rr
        outcomes$time <- 1+round(runif(n=nrow(outcomes),min=0,max=499))
        outcomes$y <- rpois(nrows,outcomes$rate)
    } else
        stop(paste("Unknown model:",model))
    
    outcomes <- outcomes[order(outcomes$stratum_id,outcomes$row_id),]
    covariates <- covariates[order(covariates$stratum_id,covariates$row_id,covariates$covariate_id),]
    sparseness <- 1-(nrow(covariates)/(nrows*ncovars))
    writeLines(paste("Sparseness =",sparseness*100,"%"))
    list(outcomes = outcomes, covariates = covariates, effectSizes = effectSizes, sparseness = sparseness)
}

fitUsingClogit <- function(sim,coverage=TRUE){
    start <- Sys.time()    
    covariates <- sim$covariates
    ncovars <- max(covariates$covariate_id)
    nrows <- nrow(sim$outcomes)
    m <- matrix(0,nrows,ncovars)
    for (i in 1:nrow(covariates)){
        m[covariates$row_id[i],covariates$covariate_id[i]] <- 1
    }
    data <- as.data.frame(m)
    
    data$row_id <- 1:nrow(data)
    data <- merge(data,sim$outcomes)
    data <- data[order(data$stratum_id,data$row_id),]
    formula <- as.formula(paste(c("y ~ strata(stratum_id)",paste("V",1:ncovars,sep="")),collapse=" + "))
    fit <- clogit(formula,data=data) 
    if (coverage) {
        ci <- confint(fit)
    } else {
        ci <- matrix(0,nrow=1,ncol=2)
    }
    
    delta <- Sys.time() - start
    writeLines(paste("Analysis took", signif(delta,3), attr(delta,"units")))
    
    data.frame(coef = coef(fit), lbCi95 = ci[,1], ubCi95 = ci[,2])
}

fitUsingCoxph <- function(sim){
    require("survival")
    start <- Sys.time()    
    covariates <- sim$covariates
    ncovars <- max(covariates$covariate_id)
    nrows <- nrow(sim$outcomes)
    m <- matrix(0,nrows,ncovars)
    for (i in 1:nrow(covariates)){
        m[covariates$row_id[i],covariates$covariate_id[i]] <- 1
    }
    data <- as.data.frame(m)
    
    data$row_id <- 1:nrow(data)
    data <- merge(data,sim$outcomes)
    data <- data[order(data$stratum_id,data$row_id),]
    formula <- as.formula(paste(c("Surv(time,y) ~ strata(stratum_id)",paste("V",1:ncovars,sep="")),collapse=" + "))
    fit <- coxph(formula,data=data)    
    if (coverage) {
        ci <- confint(fit)
    } else {
        ci <- matrix(0,nrow=1,ncol=2)
    }
    
    delta <- Sys.time() - start
    writeLines(paste("Analysis took", signif(delta,3), attr(delta,"units")))
    
    data.frame(coef = coef(fit), lbCi95 = ci[,1], ubCi95 = ci[,2])
}

fitUsingGnm <- function(sim,coverage=TRUE){
    require(gnm)
    start <- Sys.time()    
    covariates <- sim$covariates
    ncovars <- max(covariates$covariate_id)
    nrows <- nrow(sim$outcomes)
    m <- matrix(0,nrows,ncovars)
    for (i in 1:nrow(covariates)){
        m[covariates$row_id[i],covariates$covariate_id[i]] <- 1
    }
    data <- as.data.frame(m)
    
    data$row_id <- 1:nrow(data)
    data <- merge(data,sim$outcomes)
    data <- data[order(data$stratum_id,data$row_id),]
    formula <- as.formula(paste(c("y ~ V1",paste("V",2:ncovars,sep="")),collapse=" + "))
    
    fit = gnm(formula, family=poisson, offset=log(time), eliminate=as.factor(stratum_id), data = data)
    #Todo: figure out how to do confidence intervals correctly
    confint(fit)
    fit0 = gnm(y ~ 1, family=poisson, offset=log(time), eliminate=as.factor(stratum_id), data = data)
    se <- abs(coef(fit)[[1]]/qnorm(1-pchisq(deviance(fit0)-deviance(fit),1)))
    
    
    fit <- coxph(formula,data=data)    
    if (coverage) {
        ci <- confint(fit)
    } else {
        ci <- matrix(0,nrow=1,ncol=2)
    }
    
    delta <- Sys.time() - start
    writeLines(paste("Analysis took", signif(delta,3), attr(delta,"units")))
    
    data.frame(coef = coef(fit), lbCi95 = ci[,1], ubCi95 = ci[,2])
}


fitUsingOtherThanCyclops <- function(sim, model="logistic",coverage=TRUE){
    if (model == "logistic"){
        writeLines("Fitting model using clogit")
        fitUsingClogit(sim,coverage)   
    } else if (model == "poisson"){
        stop("Poisson not yet implemented")
    } else if (model == "survival"){
        writeLines("Fitting model using coxpht")
        fitUsingCoxph(sim,coverage)
    } else {
        stop(paste("Unknown model:",model))
    }
}

fitUsingCyclops <- function(sim, 
                            model = "logistic",
                            regularized = TRUE,                                                       
                            coverage = TRUE,
                            includePenalty = FALSE){
    if (!regularized)
        includePenalty = FALSE
    start <- Sys.time()    
    stratified <- max(sim$outcomes$stratum_id) > 1
    if (stratified){
        if (model == "logistic") modelType = "clr"
        if (model == "poisson") modelType = "cpr"
        if (model == "survival") modelType = "cox"
    } else {
        if (model == "logistic") modelType = "lr"
        if (model == "poisson") modelType = "pr"
        if (model == "survival") modelType = "cox"        
    }
    
    if (!stratified){
        sim$outcomes$stratum_id = sim$outcomes$row_id
        sim$outcomes <- sim$outcomes(order(sim$outcomes$stratum_id))
        sim$covariates$stratum_id = sim$covariates$row_id
        sim$covariates <- sim$covariates(order(sim$covariates$stratum_id))
    }
    if (model != "poisson" & model != "survival"){
        sim$outcomes$time <- 1
    }
    if (model == "survival"){
        sim$outcomes <- sim$outcomes[order(sim$outcomes$stratum_id,-sim$outcomes$time,sim$outcomes$y,sim$outcomes$row_id),]
        sim$covariates <- merge(sim$outcomes[,c("time","y","row_id")],sim$covariates)
        sim$covariates <- sim$covariates[order(sim$covariates$stratum_id,-sim$covariates$time,sim$covariates$y,sim$covariates$row_id),]        
    }
    
    dataPtr <- createSqlCyclopsData(modelType = modelType)
    
    count <- appendSqlCyclopsData(dataPtr,
                                  sim$outcomes$stratum_id,
                                  sim$outcomes$row_id,
                                  sim$outcomes$y,
                                  sim$outcomes$time,
                                  sim$covariates$row_id,
                                  sim$covariates$covariate_id,
                                  sim$covariates$covariate_value)
    if (model == "poisson")
        finalizeSqlCyclopsData(dataPtr,useOffsetCovariate=-1) 
    else if (model != "survival")
        finalizeSqlCyclopsData(dataPtr) 
    
    if (regularized){
        coefCyclops <- rep(0,length(sim$effectSizes$rr))
        lbCi95 <- rep(0,length(sim$effectSizes$rr))
        ubCi95 <- rep(0,length(sim$effectSizes$rr))
        for (i in 1:length(sim$effectSizes$rr)){
            if (model == "survival"){ # For some reason can't get confint twice on same dataPtr object, so recreate:
                dataPtr <- createSqlCyclopsData(modelType = modelType)
                
                count <- appendSqlCyclopsData(dataPtr,
                                              sim$outcomes$stratum_id,
                                              sim$outcomes$row_id,
                                              sim$outcomes$y,
                                              sim$outcomes$time,
                                              sim$covariates$row_id,
                                              sim$covariates$covariate_id,
                                              sim$covariates$covariate_value)
                
            }
            cyclopsFit <- fitCyclopsModel(dataPtr,forceColdStart=FALSE,prior = prior("laplace",0.1,exclude=i))
            coefCyclops[i] <- coef(cyclopsFit)[names(coef(cyclopsFit)) == as.character(i)]
            if (coverage) {
                if (model == "survival"){
                    ci <- confint(cyclopsFit,parm=i,includePenalty = includePenalty)
                    lbCi95 <- ci[,2]
                    ubCi95 <- ci[,3]
                } else {
                    ci <- aconfint(cyclopsFit,parm=i)
                    lbCi95[i] <- ci[,1]
                    ubCi95[i] <- ci[,2]
                }
            }
        }
    } else {
        cyclopsFit <- fitCyclopsModel(dataPtr, prior = prior("none"))
        coefCyclops <- data.frame(covariate_id = as.integer(names(coef(cyclopsFit))),beta=coef(cyclopsFit))
        coefCyclops <- coefCyclops[order(coefCyclops$covariate_id),]
        coefCyclops <- coefCyclops$beta
        if (coverage) {
            if (model == "survival"){
                ci <- confint(cyclopsFit,parm=i,includePenalty = includePenalty)
                lbCi95 <- ci[,2]
                ubCi95 <- ci[,3]
            } else {
                ci <- aconfint(cyclopsFit,parm=i)
                lbCi95 <- ci[,1]
                ubCi95 <- ci[,2]
            }
        }
    }
    delta <- Sys.time() - start
    writeLines(paste("Analysis took", signif(delta,3), attr(delta,"units")))
    
    df <- data.frame(coef = coefCyclops, lbCi95 = lbCi95, ubCi95 = ubCi95)
    attr(df, "dataPtr") <- dataPtr
    df
}

mse <- function(x,y){
    mean((x-y)^2)
}

coverage <- function(goldStandard,lowerBounds,upperBounds){
    sum(goldStandard >= lowerBounds & goldStandard <= upperBounds) / length(goldStandard)
}

plotFit <- function(fit,goldStandard,label){
    require("ggplot2")
    ggplot(fit, aes(x= goldStandard , y=coef, ymin=lbCi95, ymax=ubCi95), environment=environment()) +
        geom_abline(intercept = 0, slope = 1) +
        geom_pointrange(alpha=0.2) +
        scale_y_continuous(label)
}

runSimulation1 <- function(){
    model = "survival"      # "logistic", "survival", or "poisson"
    sim <- simulateData(nstrata=2000,
                        ncovars=100,
                        nrows=10000,
                        effectSizeSd=0.5,
                        eCovarsPerRow=2,
                        model=model)
    coefGoldStandard <- log(sim$effectSizes$rr)
    
    fitCyclops <- fitUsingCyclops(sim,regularized=TRUE,model)
    fit <- fitUsingOtherThanCyclops(sim,model)
    
    writeLines(paste("MSE Cyclops:",mse(fitCyclops$coef,coefGoldStandard)))
    writeLines(paste("MSE other:",mse(fit$coef,coefGoldStandard)))
    
    plotFit(fitCyclops,coefGoldStandard,"Cyclops")  
    plotFit(fit,coefGoldStandard,"Other")
    
    writeLines(paste("Coverage Cyclops:",coverage(coefGoldStandard,fitCyclops$lbCi95, fitCyclops$ubCi95)))
    writeLines(paste("Coverage other:",coverage(coefGoldStandard,fit$lbCi95, fit$ubCi95)))
}

runSimulation2 <- function(){
    model = "logistic"      # "logistic", "survival", or "poisson"
    sim <- simulateData(nstrata=2000,
                        ncovars=100,
                        nrows=10000,
                        effectSizeSd=0.5,
                        eCovarsPerRow=2,
                        model=model)
    coefGoldStandard <- log(sim$effectSizes$rr)
    
    fit <- fitUsingOtherThanCyclops(sim,model,coverage=FALSE)
    fitCyclops <- fitUsingCyclops(sim,regularized=TRUE,model,coverage=FALSE)
        
    writeLines(paste("MSE Cyclops:",mse(fitCyclops$coef,coefGoldStandard)))
    writeLines(paste("MSE other:",mse(fit$coef,coefGoldStandard)))
}