# Extract the gold-standard values the R testthat suite compares against, so the
# Python port can assert the same numbers without needing R at test time.
#
# Gold standards come from glm / lm / coxph / clogit / gnm -- NOT from Cyclops --
# which is what makes the ported tests independent validation rather than a
# self-consistency check.

suppressMessages({
    library(Cyclops); library(survival); library(gnm); library(jsonlite); library(MASS)
})

out <- list()
dir.create("fixtures", showWarnings = FALSE)

# --- 1. Bernoulli "bid" data (test-smallBernoulli.R) ------------------------
binomial_bid <- c(1,5,10,20,30,40,50,75,100,150,200)
binomial_n   <- c(31,29,27,25,23,21,19,17,15,15,15)
binomial_y   <- c(0,3,6,7,9,13,17,12,11,14,13)
log_bid <- log(c(rep(binomial_bid, binomial_n - binomial_y), rep(binomial_bid, binomial_y)))
y <- c(rep(0, sum(binomial_n - binomial_y)), rep(1, sum(binomial_y)))

glmFit <- glm(y ~ log_bid, family = binomial())
out$bernoulli <- list(
    n = length(y),
    coef = as.numeric(coef(glmFit)),
    log_likelihood = as.numeric(logLik(glmFit)),
    confint = as.numeric(t(suppressMessages(confint(glmFit)))),   # row-major (lo,hi) pairs
    predict_head = as.numeric(head(predict(glmFit, type = "response"), 5))
)
write.csv(data.frame(y = y, log_bid = log_bid), "fixtures/bernoulli.csv", row.names = FALSE)

# --- 2. Dobson Poisson (test-smallPoisson.R) --------------------------------
dobson <- data.frame(counts = c(18,17,15,20,10,20,25,13,12),
                     outcome = gl(3,1,9), treatment = gl(3,3))
gold <- glm(counts ~ outcome + treatment, data = dobson, family = poisson())
mm <- model.matrix(~ outcome + treatment, data = dobson)   # (Intercept), outcome2/3, treatment2/3
out$poisson <- list(
    design = as.numeric(t(mm[, -1])),      # row-major, intercept dropped
    n_features = ncol(mm) - 1,
    counts = as.numeric(dobson$counts),
    coef = as.numeric(coef(gold)),
    log_likelihood = as.numeric(logLik(gold)),
    confint = as.numeric(t(suppressMessages(confint(gold)))),
    predict = as.numeric(predict(gold, type = "response")),
    # univariable correlations (test-correlation.R): NA for the intercept
    correlation = as.numeric(sapply(2:5, function(i) cor(dobson$counts, mm[, i])))
)

# Poisson with an offset (test-smallPoisson.R "with offset").
# `outcome` enters as a numeric offset on the log scale.
dobson2 <- data.frame(counts = c(18,17,15,20,10,20,25,13,12),
                      outcome = as.numeric(gl(3,1,9)), treatment = gl(3,3))
goldOff <- glm(counts ~ treatment, offset = outcome, data = dobson2, family = poisson())
mmOff <- model.matrix(~ treatment, data = dobson2)
out$poisson_offset <- list(
    design = as.numeric(t(mmOff[, -1])),
    n_features = ncol(mmOff) - 1,
    counts = as.numeric(dobson2$counts),
    log_offset = as.numeric(dobson2$outcome),
    coef = as.numeric(coef(goldOff)),
    log_likelihood = as.numeric(logLik(goldOff))
)

# --- 3. Small Normal / least squares (test-smallNormal.R) -------------------
x <- log(c(1,5,10,20,30,40,50,75,100,150,200))
yn <- c(0,3,6,7,9,13,17,12,11,14,13)
lmFit <- lm(yn ~ x)
out$normal <- list(x = as.numeric(x), y = as.numeric(yn),
                   coef = as.numeric(coef(lmFit)))

# --- 4. Small Cox examples (test-smallCox.R) --------------------------------
coxCase <- function(text, label) {
    test <- read.table(header = TRUE, sep = ",", text = text)
    right <- coxph(Surv(length, event) ~ x1 + x2, test)
    strat <- coxph(Surv(length, event) ~ x1 + strata(x2), test)
    list(
        length = as.numeric(test$length), event = as.numeric(test$event),
        x1 = as.numeric(test$x1), x2 = as.numeric(test$x2),
        coef = as.numeric(coef(right)),
        log_likelihood = as.numeric(right$loglik[2]),
        se = as.numeric(sqrt(diag(vcov(right)))),
        strat_coef = as.numeric(coef(strat)),
        strat_log_likelihood = as.numeric(strat$loglik[2])
    )
}
out$cox_no_ties <- coxCase("
start, length, event, x1, x2
0, 4,  1,0,0
0, 3.5,1,2,0
0, 3,  0,0,1
0, 2.5,1,0,1
0, 2,  1,1,1
0, 1.5,0,1,0
0, 1,  1,1,0
")
out$cox_time_ties <- coxCase("
start, length, event, x1, x2
0, 4,  1,0,0
0, 3,  1,2,0
0, 3,  0,0,1
0, 2,  1,0,1
0, 2,  0,1,1
0, 1.5,0,1,0
0, 1,  1,1,0
")

# --- 5. Conditional logistic on `infert` (test-smallCLR.R) ------------------
goldClr <- clogit(case ~ spontaneous + induced + strata(stratum), data = infert)
out$clr <- list(
    coef = as.numeric(coef(goldClr)),
    log_likelihood = as.numeric(logLik(goldClr)),
    se = as.numeric(sqrt(diag(vcov(goldClr)))),
    confint = as.numeric(t(confint(goldClr)))
)
write.csv(infert[, c("stratum", "case", "spontaneous", "induced")],
          "fixtures/infert.csv", row.names = FALSE)

# --- 6. Conditional Poisson on `oxford` (test-smallCLR.R / conditionalPoisson) --
goldCpr <- gnm(event ~ exgr + agegr + offset(loginterval), family = poisson,
               eliminate = indiv, data = Cyclops::oxford)
out$cpr <- list(
    coef = as.numeric(coef(goldCpr)),
    se = as.numeric(sqrt(diag(vcov(goldCpr))))
)
# SCCS reads person-time as its `time` vector, so keep the untransformed interval.
goldSccs <- clogit(event ~ exgr + agegr + strata(indiv) + offset(loginterval),
                   data = Cyclops::oxford)
out$sccs <- list(coef = as.numeric(coef(goldSccs)),
                 log_likelihood = as.numeric(logLik(goldSccs)))
write.csv(Cyclops::oxford[, c("indiv", "event", "interval", "agegr", "exgr")],
          "fixtures/oxford.csv", row.names = FALSE)

# --- 7. Cox on `bladder` (test-gradient.R) ----------------------------------
# Cyclops uses Breslow tie handling; coxph defaults to Efron, and `bladder` has
# many tied event times, so the default would not be a like-for-like gold.
bladderGold <- coxph(Surv(stop, event) ~ rx + size, data = survival::bladder,
                     ties = "breslow")
out$bladder <- list(coef = as.numeric(coef(bladderGold)),
                    log_likelihood = as.numeric(bladderGold$loglik[2]))
write.csv(survival::bladder[, c("rx", "size", "stop", "event")],
          "fixtures/bladder.csv", row.names = FALSE)

writeLines(toJSON(out, digits = 17, auto_unbox = TRUE), "fixtures/gold.json")
cat("wrote fixtures/gold.json and CSVs\n")
