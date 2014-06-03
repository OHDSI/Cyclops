#' @title createCcdDataFrame
#'
#' @description
#' \code{createCcdDataFrame} creates a CCD model data object from an R formula
#'
#' @details
#' This function creates a CCD model data object from an R formula and data.frame.
#'
#' @param formula			An R formula
#' @param data
#' 
#' @return
#' A list that contains a CCD model data object pointer and an operation duration
#' 
#' @examples
#' ## Dobson (1990) Page 93: Randomized Controlled Trial :
#' counts <- c(18,17,15,20,10,20,25,13,12)
#' outcome <- gl(3,1,9)
#' treatment <- gl(3,3)
#' ccdModelData <- createCcdDataFrame(counts ~ outcome + treatment, model=TRUE)
#' ccdModel <- ccdInitializeModel(ccdModelData$ccdModelDataPtr, modelType="pr", computeMLE=TRUE)
#' ccdFit <- ccdFitModel(ccdModel$interface)
#' ccdLog <- ccdLogModel(ccdModel$interface)
#' as.data.frame(ccdLog$estimation)
#'
#'
#' ccdModelData2 <- createCcdDataFrame(counts ~ outcome, indicatorFormula = ~ treatment, model=TRUE)
#' ccdModel2 <- ccdInitializeModel(ccdModelData2$ccdModelDataPtr, modelType="pr", computeMLE=TRUE)
#' ccdFit2 <- ccdFitModel(ccdModel2$interface)
#' ccdLog2 <- ccdLogModel(ccdModel2$interface)
#' as.data.frame(ccdLog2$estimation)
#'
#' @export
createCcdDataFrame <- function(formula, sparseFormula, indicatorFormula,
	data, subset, weights, offs = NULL, y = NULL, z = NULL, dx = NULL, 
	sx = NULL, ix = NULL, model = FALSE, method = "ccd.fit", ...) {	
	cl <- match.call() # save to return
	mf.all <- match.call(expand.dots = FALSE)
			
	if (!missing(formula)) { # Use formula to construct CCD matrices
		if (missing(data)) {
			data <- environment(formula)
		}					
		m.d <- match(c("formula", "data", "subset", "weights",
				"offset"), names(mf.all), 0L)
		mf.d <- mf.all[c(1L, m.d)]
		mf.d$drop.unused.levels <- TRUE
		mf.d[[1L]] <- quote(stats::model.frame)		
		mf.d <- eval(mf.d, parent.frame())
		dx <- Matrix(model.matrix(mf.d), sparse=FALSE)	# TODO sparse / indicator matrices
		
		y <- model.response(mf.d) # TODO Update for censored outcomes
		pid <- c(1:length(y)) # TODO Update for stratified models		       
    
		if (!missing(sparseFormula)) {					
			if (missing(data)) {
				data <- environment(sparseFormula)
			}					
			m.s <- match(c("sparseFormula", "data", "subset", "weights",
					"offset"), names(mf.all), 0L)
			mf.s <- mf.all[c(1L, m.s)]
			mf.s$drop.unused.levels <- TRUE
			mf.s[[1L]] <- quote(stats::model.frame)
			mf.s[[2L]] <- update(mf.s[[2L]], ~ . - 1) # Remove intercept
			names(mf.s)[2] = "formula"
			mf.s <- eval(mf.s, parent.frame())
			attr(attr(mf.s, "terms"), "intercept") <- 0  # Remove intercept
			
			if (!is.null(model.response(mf.s))) {
				stop("Must only provide outcome variable in dense formula.")
			}
			
			if (attr(attr(mf.s, "terms"), "intercept") == 1) { # Remove intercept
				sx <- Matrix(model.matrix(mf.s), sparse=TRUE)[,-1]
			} else {
				sx <- Matrix(model.matrix(mf.s), sparse=TRUE)			
			}					
		}    
	
		if (!missing(indicatorFormula)) {
			if (missing(data)) {
				data <- environment(indicatorFormula)
			}						
			m.i <- match(c("indicatorFormula", "data", "subset", "weights",
					"offset"), names(mf.all), 0L)
			mf.i <- mf.all[c(1L, m.i)]
			mf.i$drop.unused.levels <- TRUE
			mf.i[[1L]] <- quote(stats::model.frame)
			names(mf.i)[2] = "formula"				
			mf.i <- eval(mf.i, parent.frame())
			hasIntercept <- attr(attr(mf.i, "terms"), "intercept") == 1
			
			if (!is.null(model.response(mf.i))) {
				stop("Must only provide outcome variable in dense formula.")
			}			
			
			if (attr(attr(mf.i, "terms"), "intercept") == 1) { # Remove intercept
				ix <- Matrix(model.matrix(mf.i), sparse=TRUE)[,-1]
			} else {
				ix <- Matrix(model.matrix(mf.i), sparse=TRUE)			
			}
		} 
		
		if (identical(method, "model.frame")) {
			result <- list()
			if (exists("mf.d")) {
				result$dense <- mf.d
				result$dx <- dx
			}
			if (exists("mf.s")) {
				result$sparse <- mf.s
				result$sx <- sx
			}
			if (exists("mf.i")) {
				result$indicator <- mf.i
				result$ix <- ix
			}	
			return(result)    	
		}     		
	} else  {
		if (!missing(sparseFormula) || !missing(indicatorFormula)) {
			stop("Must provide a dense formula when specifying sparse or indicator covariates.")
		}	
	}
    
 
    
    # TODO Check types and dimensions        
    
    md <- ccdModelData(pid, y, z, offs, dx, sx, ix)
	result <- list()
	result$ccdModelDataPtr <- md$data
	result$timeLoad <- md$timeLoad
	result$call <- cl
	if (exists("mf") && model == TRUE) {
		result$mf <- mf
	}
	result$dx <- dx
	result$sx <- sx
	result$ix <- ix
	result$y <- y
	result$pid <- pid
	return(result)
}

glm.bad <-
function (formula, family = gaussian, data, weights, subset, 
    na.action, start = NULL, etastart, mustart, offset, control = list(...), 
    model = TRUE, method = "glm.fit", x = FALSE, y = TRUE, contrasts = NULL, 
    ...) 
{
    call <- match.call()
    if (is.character(family)) 
        family <- get(family, mode = "function", envir = parent.frame())
    if (is.function(family)) 
        family <- family()
    if (is.null(family$family)) {
        print(family)
        stop("'family' not recognized")
    }
    if (missing(data)) 
        data <- environment(formula)
    mf <- match.call(expand.dots = FALSE)
    m <- match(c("formula", "data", "subset", "weights", "na.action", 
        "etastart", "mustart", "offset"), names(mf), 0L)
    mf <- mf[c(1L, m)]
    mf$drop.unused.levels <- TRUE
    mf[[1L]] <- quote(stats::model.frame)
    mf <- eval(mf, parent.frame())
    if (identical(method, "model.frame")) 
        return(mf)
    if (!is.character(method) && !is.function(method)) 
        stop("invalid 'method' argument")
    if (identical(method, "glm.fit")) 
        control <- do.call("glm.control", control)
    mt <- attr(mf, "terms")
    Y <- model.response(mf, "any")
    if (length(dim(Y)) == 1L) {
        nm <- rownames(Y)
        dim(Y) <- NULL
        if (!is.null(nm)) 
            names(Y) <- nm
    }
    X <- if (!is.empty.model(mt)) 
        model.matrix(mt, mf, contrasts)
    else matrix(, NROW(Y), 0L)
    weights <- as.vector(model.weights(mf))
    if (!is.null(weights) && !is.numeric(weights)) 
        stop("'weights' must be a numeric vector")
    if (!is.null(weights) && any(weights < 0)) 
        stop("negative weights not allowed")
    offset <- as.vector(model.offset(mf))
    if (!is.null(offset)) {
        if (length(offset) != NROW(Y)) 
            stop(gettextf("number of offsets is %d should equal %d (number of observations)", 
                length(offset), NROW(Y)), domain = NA)
    }
    mustart <- model.extract(mf, "mustart")
    etastart <- model.extract(mf, "etastart")
    fit <- eval(call(if (is.function(method)) "method" else method, 
        x = X, y = Y, weights = weights, start = start, etastart = etastart, 
        mustart = mustart, offset = offset, family = family, 
        control = control, intercept = attr(mt, "intercept") > 
            0L))
    if (length(offset) && attr(mt, "intercept") > 0L) {
        fit2 <- eval(call(if (is.function(method)) "method" else method, 
            x = X[, "(Intercept)", drop = FALSE], y = Y, weights = weights, 
            offset = offset, family = family, control = control, 
            intercept = TRUE))
        if (!fit2$converged) 
            warning("fitting to calculate the null deviance did not converge -- increase 'maxit'?")
        fit$null.deviance <- fit2$deviance
    }
    if (model) 
        fit$model <- mf
    fit$na.action <- attr(mf, "na.action")
    if (x) 
        fit$x <- X
    if (!y) 
        fit$y <- NULL
    fit <- c(fit, list(call = call, formula = formula, terms = mt, 
        data = data, offset = offset, control = control, method = method, 
        contrasts = attr(X, "contrasts"), xlevels = .getXlevels(mt, 
            mf)))
    class(fit) <- c(fit$class, c("glm", "lm"))
    fit
}


lm.bad <-
function (formula, data, subset, weights, na.action, method = "qr", 
    model = TRUE, x = FALSE, y = FALSE, qr = TRUE, singular.ok = TRUE, 
    contrasts = NULL, offset, ...) 
{
    ret.x <- x
    ret.y <- y
    cl <- match.call()
    mf <- match.call(expand.dots = FALSE)
    m <- match(c("formula", "data", "subset", "weights", "na.action", 
        "offset"), names(mf), 0L)
    mf <- mf[c(1L, m)]
    mf$drop.unused.levels <- TRUE
    mf[[1L]] <- quote(stats::model.frame)
    mf <- eval(mf, parent.frame())
    if (method == "model.frame") 
        return(mf)
    else if (method != "qr") 
        warning(gettextf("method = '%s' is not supported. Using 'qr'", 
            method), domain = NA)
    mt <- attr(mf, "terms")
    y <- model.response(mf, "numeric")
    w <- as.vector(model.weights(mf))
    if (!is.null(w) && !is.numeric(w)) 
        stop("'weights' must be a numeric vector")
    offset <- as.vector(model.offset(mf))
    if (!is.null(offset)) {
        if (length(offset) != NROW(y)) 
            stop(gettextf("number of offsets is %d, should equal %d (number of observations)", 
                length(offset), NROW(y)), domain = NA)
    }
    if (is.empty.model(mt)) {
        x <- NULL
        z <- list(coefficients = if (is.matrix(y)) matrix(, 0, 
            3) else numeric(), residuals = y, fitted.values = 0 * 
            y, weights = w, rank = 0L, df.residual = if (!is.null(w)) sum(w != 
            0) else if (is.matrix(y)) nrow(y) else length(y))
        if (!is.null(offset)) {
            z$fitted.values <- offset
            z$residuals <- y - offset
        }
    }
    else {
        x <- model.matrix(mt, mf, contrasts)
        z <- if (is.null(w)) 
            lm.fit(x, y, offset = offset, singular.ok = singular.ok, 
                ...)
        else lm.wfit(x, y, w, offset = offset, singular.ok = singular.ok, 
            ...)
    }
    class(z) <- c(if (is.matrix(y)) "mlm", "lm")
    z$na.action <- attr(mf, "na.action")
    z$offset <- offset
    z$contrasts <- attr(x, "contrasts")
    z$xlevels <- .getXlevels(mt, mf)
    z$call <- cl
    z$terms <- mt
    if (model) 
        z$model <- mf
    if (ret.x) 
        z$x <- x
    if (ret.y) 
        z$y <- y
    if (!qr) 
        z$qr <- NULL
    z
}