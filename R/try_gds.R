library(bsccs)
library(bayesGDS)

library(Matrix)
library(foreach)
library(doParallel)
library(plyr)
#library(trustOptim)
#library(bayesGDS)

run.par <- FALSE
if(run.par) registerDoParallel(cores=12) else registerDoParallel(cores=1)

check.phi <- TRUE

set.seed(42)

data.loaded = FALSE

infer.hyperprior = TRUE
initial.hyperprior = 100

if(data.loaded == FALSE) {
    cat("Loading data...\n")
#    model = bsccs_load("short.txt")
#    model = bsccs_load("cond_500000503_sccs_multivar.txt")    
#    model = bsccs_load("cond_500000501_sccs_multivar.txt")  
#    model = bsccs_load("cond_500000403_sccs_multivar.txt")  
    model= bsccs_load("infert.txt","csv")
    bsccs_set_hyperprior(model, initial.hyperprior)
    data.loaded = TRUE
}

cat("Finding mode...\n")
mode = bsccs_find_mode(model, infer.hyperprior=infer.hyperprior)


get.log.post.model = function(x, model, infer.hyperprior = FALSE) {
    if (infer.hyperprior == TRUE) {
        dim = length(x)
        bsccs_set_beta(model, x[1:(dim-1)])
        hyperprior = exp(x[dim])
        bsccs_set_hyperprior(model, hyperprior)        
    } else {        
        bsccs_set_beta(model, x)
    }
    f = bsccs_get_log_likelihood(model) + bsccs_get_log_prior(model)
    return(f)
}

cat("Ready to start...\n")
#browser()

## Section C:  Set parameters for the sampling algorithm
n.draws <- 1000
M <- 3000
N <- 3000
#M = 10
#N = 10
max.AR.tries <- 25000
#ds.scale <- 20
ds.scale = 2
get.log.prop <- get.log.dens.MVN
draw.prop <- draw.MVN.proposals


##
dim = length(mode$beta)
if (infer.hyperprior == TRUE) {
    dim = dim + 1
}
chol.prec = t(chol(ds.scale*diag(dim)))
post.mode = mode$beta
if (infer.hyperprior == TRUE) {
    post.mode = c(post.mode, log(mode$hyperprior))
}

## Section G:  The Generalized Direct Sampler

prop.params <- list(mu = post.mode,
                    chol.prec = chol.prec
                    )

log.c1 <- mode$logLike + mode$logPrior
log.c2 <- get.log.prop(post.mode, prop.params)
log.const <- log.c1 - log.c2

cat("Collecting GDS Proposal Draws\n")
draws.m <- foreach(mz=1:M,
                   .inorder=FALSE) %dopar% draw.prop(n=1, params=as(prop.params,"vector"))
                   
##  compute log posterior density for all proposals
cat("Collecting log posterior evaluations\n")
log.post.m <- laply(draws.m, get.log.post.model,model=model, infer.hyperprior=infer.hyperprior, .parallel=run.par)

## drop any proposals with Inf or NaN posteriors (numerical underflow issues?)
wf <- is.finite(log.post.m)
draws.m <- draws.m[wf]
log.post.m <- log.post.m[wf]
M <- sum(wf)

## log proposal densities for all proposals
log.prop.m <- laply(draws.m, get.log.prop, params=prop.params)
log.phi <- log.post.m - log.prop.m + log.c2 - log.c1

cat("Are any log.phi > 0?  ",any(log.phi>0),"\n")

if (check.phi == TRUE) {
    browser()
}

Z <- median(log.phi) ## median log.phi
q <- get.bernstein.weights(log.phi, N, Z)

cat("Generating GDS draws - accept-reject phase\n")
draws <- get.GDS.draws(n.draws, q, log.const, Z,
                       get.log.post.model,
                       draw.prop,
                       get.log.prop,
                       prop.params,
                       max.tries=max.AR.tries,
                       model=model,
                       infer.hyperprior = infer.hyperprior,
                       run.par=run.par,
                       .progress="text")