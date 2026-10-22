# Reference prototype and validation for the counting-process Cox engine ("cox_counting").
#
# The engine keeps the existing Cox data layout -- rows sorted by (stratum, -stop, event) --
# and gives every row two positions in the list of risk-set groups (the distinct event times
# of its stratum, in scan order):
#   entry = first event time <= stop   (the row joins the risk set here)
#   exit  = first event time <= start  (the row has left the risk set here)
# Adding a row's contribution at `entry` and subtracting it at `exit` turns every risk-set sum
# into one forward prefix scan over the groups, so the log-likelihood, gradient and Hessian
# stay O(K + N) per evaluation (K rows, N event times).
#
# Run with: Rscript extras/CountingProcessCoxPrototype.R

library(survival)

# ---- one-time layout, O(K log N) ----

cpLayout <- function(start, stop, event, stratum, weight) {
    K <- length(stop)
    entry <- exit <- rep(NA_integer_, K)
    groupStratum <- integer(0)
    N <- 0L
    for (s in unique(stratum)) {
        rows <- which(stratum == s & weight != 0)
        times <- unique(stop[rows][event[rows] != 0]) # strictly decreasing (rows are sorted)
        if (length(times) == 0) next
        ascending <- rev(times)
        position <- function(x) { # first group (in scan order) whose time <= x
            n <- findInterval(x, ascending)
            ifelse(n == 0, NA_integer_, N + length(times) - n + 1L)
        }
        en <- position(stop[rows])
        ex <- position(start[rows])
        never <- is.na(en) | (!is.na(ex) & ex == en) # never at risk at an event time
        entry[rows[!never]] <- en[!never]
        exit[rows[!never]] <- ex[!never]
        groupStratum <- c(groupStratum, rep(s, length(times)))
        N <- N + length(times)
    }
    isEvent <- event != 0 & !is.na(entry)
    list(entry = entry, exit = exit, N = N, groupStratum = groupStratum,
         events = groupSum(entry[isEvent], weight[isEvent], N))
}

groupSum <- function(group, value, N) {
    out <- numeric(N)
    if (length(group) > 0) {
        s <- rowsum(value, group)
        out[as.integer(rownames(s))] <- s
    }
    out
}

# Segmented prefix scan: risk-set sum of `value` at every event-time group
cpScan <- function(value, layout) {
    on <- !is.na(layout$entry)
    off <- !is.na(layout$exit)
    delta <- groupSum(layout$entry[on], value[on], layout$N) -
        groupSum(layout$exit[off], value[off], layout$N)
    ave(delta, layout$groupStratum, FUN = cumsum)
}

# Log-likelihood, gradient and Hessian diagonal of the *negative* log-likelihood, as in Cyclops
cpDerivatives <- function(beta, X, data, layout) {
    eta <- drop(X %*% beta)
    u <- data$weight * exp(eta)
    D <- cpScan(u, layout)
    d <- layout$events
    wEvent <- data$weight * data$event
    grad <- hess <- numeric(ncol(X))
    for (j in seq_len(ncol(X))) {
        N1 <- cpScan(u * X[, j], layout)
        N2 <- cpScan(u * X[, j]^2, layout)
        grad[j] <- sum(d * N1 / D) - sum(wEvent * X[, j])
        hess[j] <- sum(d * (N2 / D - (N1 / D)^2))
    }
    list(loglik = sum(wEvent * eta) - sum(d * log(D)), gradient = grad, hessian = hess)
}

# Brute-force O(K * N) reference: enumerate each risk set directly
bruteDerivatives <- function(beta, X, data) {
    eta <- drop(X %*% beta)
    u <- data$weight * exp(eta)
    wEvent <- data$weight * data$event
    loglik <- sum(wEvent * eta)
    grad <- -colSums(X * wEvent)
    hess <- numeric(ncol(X))
    for (s in unique(data$stratum)) {
        inS <- data$stratum == s & data$weight != 0
        for (t in unique(data$stop[inS & data$event != 0])) {
            risk <- inS & data$start < t & data$stop >= t
            d <- sum(wEvent[inS & data$stop == t])
            D <- sum(u[risk])
            N1 <- colSums(X[risk, , drop = FALSE] * u[risk])
            N2 <- colSums(X[risk, , drop = FALSE]^2 * u[risk])
            loglik <- loglik - d * log(D)
            grad <- grad + d * N1 / D
            hess <- hess + d * (N2 / D - (N1 / D)^2)
        }
    }
    list(loglik = loglik, gradient = grad, hessian = hess)
}

# Candidate optimization: gradient/Hessian of a sparse column in O(#entry/exit points).
# Between consecutive entry/exit points of column j the numerators are constant, so each
# segment contributes numerator * (range sum of d/D) and numerator^2 * (range sum of d/D^2).
cpSparseDerivative <- function(j, beta, X, data, layout) {
    eta <- drop(X %*% beta)
    u <- data$weight * exp(eta)
    D <- cpScan(u, layout)
    d <- layout$events
    C <- c(0, cumsum(d / D))   # C[g + 1] = sum_{g' <= g} d / D
    P <- c(0, cumsum(d / D^2))
    stratumEnd <- cumsum(rle(layout$groupStratum)$lengths)[
        rep(seq_along(rle(layout$groupStratum)$lengths), rle(layout$groupStratum)$lengths)]
    x <- X[, j]
    nz <- which(x != 0 & !is.na(layout$entry))
    points <- sort(unique(c(layout$entry[nz], layout$exit[nz][!is.na(layout$exit[nz])])))
    on <- nz
    off <- nz[!is.na(layout$exit[nz])]
    delta1 <- groupSum(layout$entry[on], (u * x)[on], layout$N) -
        groupSum(layout$exit[off], (u * x)[off], layout$N)
    delta2 <- groupSum(layout$entry[on], (u * x^2)[on], layout$N) -
        groupSum(layout$exit[off], (u * x^2)[off], layout$N)
    grad <- -sum((data$weight * data$event * x))
    hess <- 0
    acc1 <- acc2 <- 0
    for (p in seq_along(points)) {
        a <- points[p]
        if (p > 1 && layout$groupStratum[a] != layout$groupStratum[points[p - 1]]) {
            acc1 <- acc2 <- 0 # new stratum
        }
        acc1 <- acc1 + delta1[a]
        acc2 <- acc2 + delta2[a]
        b <- min(if (p < length(points)) points[p + 1] - 1 else layout$N, stratumEnd[a])
        grad <- grad + acc1 * (C[b + 1] - C[a])
        hess <- hess + acc2 * (C[b + 1] - C[a]) - acc1^2 * (P[b + 1] - P[a])
    }
    list(gradient = grad, hessian = hess, points = length(points),
         scanLength = layout$N - min(layout$entry[nz]) + 1)
}

# ---- survival::coxph references ----

coxphAt <- function(beta, X, data) {
    fit <- suppressWarnings(coxph(Surv(start, stop, event) ~ X + strata(stratum),
                                  data = data, weights = data$weight, ties = "breslow",
                                  init = beta, control = coxph.control(iter.max = 0),
                                  subset = data$weight != 0))
    detail <- coxph.detail(fit)
    p <- ncol(X)
    list(loglik = fit$loglik[1],
         gradient = -colSums(matrix(detail$score, ncol = p)),
         hessian = diag(apply(array(detail$imat, c(p, p, length(detail$time))), c(1, 2), sum)))
}

# Newton-Raphson with the full Hessian; cross terms use the same scan
cpFit <- function(X, data, layout, iterations = 25) {
    beta <- rep(0, ncol(X))
    d <- layout$events
    for (iteration in seq_len(iterations)) {
        u <- data$weight * exp(drop(X %*% beta))
        D <- cpScan(u, layout)
        N1 <- sapply(seq_len(ncol(X)), function(j) cpScan(u * X[, j], layout))
        H <- matrix(0, ncol(X), ncol(X))
        for (j in seq_len(ncol(X))) for (k in seq_len(j)) {
            H[j, k] <- H[k, j] <- sum(d * (cpScan(u * X[, j] * X[, k], layout) / D -
                                             N1[, j] * N1[, k] / D^2))
        }
        step <- solve(H, cpDerivatives(beta, X, data, layout)$gradient)
        beta <- beta - step
        if (max(abs(step)) < 1e-13) break
    }
    beta
}

sortRows <- function(data) data[order(data$stratum, -data$stop, data$event), ]

# ---- examples ----

examples <- list()

# A: testSmallCox with delayed entry; a start equal to an event time is not at risk then
examples$delayedEntry <- read.table(header = TRUE, sep = ",", text = "
start, stop, event, x1, x2
2.0, 4.0, 1, 0, 0
1.0, 3.5, 1, 2, 0
0.5, 3.0, 0, 0, 1
0.0, 2.5, 1, 0, 1
0.0, 2.0, 1, 1, 1
1.0, 1.5, 0, 1, 0
0.0, 1.0, 1, 1, 0
")

# B: failure ties, censoring tied with events, starts tied with event times
examples$ties <- read.table(header = TRUE, sep = ",", text = "
start, stop, event, x1, x2
0.0, 4.0, 1, 0, 0
1.0, 3.0, 1, 2, 0
0.0, 3.0, 0, 0, 1
2.0, 3.0, 1, 1, 1
0.0, 2.0, 1, 0, 1
2.0, 2.5, 0, 1, 0
0.0, 2.0, 1, 1, 1
0.0, 1.0, 0, 1, 0
0.5, 1.0, 1, 1, 0
0.0, 3.0, 1, 1, 0
")

# C: strata
examples$strata <- transform(examples$ties, stratum = rep(1:2, length.out = nrow(examples$ties)))

# D: case weights, including a zero (held-out) weight
examples$weights <- transform(examples$ties, weight = c(1, 2, 0.5, 1, 1.5, 1, 0, 1, 2, 0.7))

# E: right-censored data split into counting-process rows (subjects span several rows)
lung2 <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])
lung2$status <- lung2$status - 1
lung2$id <- seq_len(nrow(lung2))
split <- survSplit(Surv(time, status) ~ ., data = lung2, cut = c(100, 200, 365, 500),
                   start = "start", end = "stop", event = "event")
examples$survSplit <- data.frame(start = split$start, stop = split$stop, event = split$event,
                                 x1 = split$age / 10, x2 = split$sex, x3 = split$ph.ecog)

# F: a genuinely time-varying covariate (switches on at a random time), with strata
set.seed(123)
n <- 400
x <- rnorm(n)
switchTime <- rexp(n, 0.5)
stratum <- sample(1:3, n, replace = TRUE)
eventTime <- numeric(n)
for (i in 1:n) { # piecewise-exponential draw with hazard exp(0.5 x) before / exp(0.5 x + 0.8) after
    t1 <- rexp(1, exp(0.5 * x[i]) * stratum[i] / 2)
    eventTime[i] <- if (t1 < switchTime[i]) t1 else
        switchTime[i] + rexp(1, exp(0.5 * x[i] + 0.8) * stratum[i] / 2)
}
censorTime <- rexp(n, 0.3)
stopTime <- pmin(eventTime, censorTime)
status <- as.numeric(eventTime <= censorTime)
before <- data.frame(start = 0, stop = pmin(stopTime, switchTime),
                     event = ifelse(stopTime <= switchTime, status, 0),
                     x1 = x, x2 = 0, stratum = stratum)
after <- data.frame(start = switchTime, stop = stopTime, event = status,
                    x1 = x, x2 = 1, stratum = stratum)[stopTime > switchTime, ]
examples$timeVarying <- rbind(before, after)

# G: left truncation with strata and case weights (larger)
set.seed(456)
n <- 3000
x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.3); x3 <- runif(n)
entryTime <- runif(n, 0, 2)
eventTime <- rexp(n, exp(0.4 * x1 - 0.6 * x2 + 0.3 * x3))
censorTime <- entryTime + rexp(n, 0.2)
keep <- eventTime > entryTime # only subjects still event-free at entry are observed
examples$leftTruncation <- data.frame(start = entryTime, stop = pmin(eventTime, censorTime),
                                      event = as.numeric(eventTime <= censorTime),
                                      x1 = x1, x2 = x2, x3 = x3,
                                      stratum = sample(1:4, n, replace = TRUE),
                                      weight = rgamma(n, 2, 2))[keep, ]

# H: start = 0 everywhere reduces to right-censored Cox
examples$startZero <- transform(examples$leftTruncation, start = 0)

# ---- validation ----

results <- lapply(names(examples), function(name) {
    data <- examples[[name]]
    if (is.null(data$stratum)) data$stratum <- 1
    if (is.null(data$weight)) data$weight <- 1
    data <- sortRows(data)
    X <- as.matrix(data[, grep("^x", names(data))])
    layout <- cpLayout(data$start, data$stop, data$event, data$stratum, data$weight)

    beta <- seq(0.2, -0.2, length.out = ncol(X))
    scan <- cpDerivatives(beta, X, data, layout)
    brute <- bruteDerivatives(beta, X, data)
    gold <- coxphAt(beta, X, data)

    goldFit <- coxph(Surv(start, stop, event) ~ X + strata(stratum), data = data,
                     weights = data$weight, ties = "breslow", subset = data$weight != 0,
                     control = coxph.control(eps = 1e-12, toler.chol = 1e-15,
                                           iter.max = 100))
    sparse <- cpSparseDerivative(1, beta, X, data, layout)

    diff <- function(a, b) max(abs(unlist(a) - unlist(b)) / pmax(1, abs(unlist(b))))
    data.frame(example = name, rows = nrow(data), eventTimes = layout$N,
               scanVsBrute = diff(scan, brute),
               scanVsCoxph = diff(scan, gold),
               fitVsCoxph = diff(cpFit(X, data, layout), coef(goldFit)),
               sparseVsScan = diff(sparse[c("gradient", "hessian")],
                                   list(scan$gradient[1], scan$hessian[1])))
})

# Invariance: splitting right-censored rows must not change the likelihood
data <- sortRows(transform(examples$survSplit, stratum = 1, weight = 1))
X <- as.matrix(data[, c("x1", "x2", "x3")])
unsplit <- sortRows(data.frame(start = 0, stop = lung2$time, event = lung2$status,
                               x1 = lung2$age / 10, x2 = lung2$sex, x3 = lung2$ph.ecog,
                               stratum = 1, weight = 1))
beta <- c(0.1, -0.5, 0.4)
splitValue <- cpDerivatives(beta, X, data, cpLayout(data$start, data$stop, data$event, 1, 1))
unsplitValue <- cpDerivatives(beta, as.matrix(unsplit[, c("x1", "x2", "x3")]), unsplit,
                              cpLayout(unsplit$start, unsplit$stop, unsplit$event, 1, 1))

results <- do.call(rbind, results)
print(format(results, digits = 3), row.names = FALSE)
cat("survSplit invariance (max abs diff):",
    format(max(abs(unlist(splitValue) - unlist(unsplitValue))), digits = 3), "\n")
stopifnot(all(results[, 4:7] < 1e-8),
          max(abs(unlist(splitValue) - unlist(unsplitValue))) < 1e-10)
cat("All counting-process prototype checks passed.\n")
