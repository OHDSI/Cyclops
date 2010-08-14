# CCD procedure for Bayesian Multiple SCCS

args = commandArgs(trailing=TRUE)

# input arguments:
	datafile = args[1]
	prior = args[2]

	if (prior != "normal" & prior != "laplace") {
		cat("Invalid prior (arg 2): must be 'normal' or 'laplace' \n")
	}

	prior_var = as.numeric(args[3])

	cat(args, "\n\n")

# avoid printing numbers in scientific notation
	options(scipen=999)

# functions ------------------------------------------------------------------

	# multivariate form for sccs likelihood, takes sparse matrix X as input
	#	where X represents a matrix of size (total numpds x num drugs)
		l = function(par, X) {
	
			X_par = X %*% par	
			exp_X_par = exp(X_par)
			offs_exp_X_par = Matrix(offs * exp_X_par)
	
			denom_pid = t(pid_mat) %*% offs_exp_X_par
	
			loglik = as.numeric( (t(y) %*% X_par) - (t(n_pid) %*% log(denom_pid)) )

			return( loglik )
		}
	
	
	# calculate gradient of log likelihood
	#		par = vector of parameters
	#		X = SPARSE matrix of covariates (sum_i(r_i) x p dimensional)
	#			where r_i = num of risk periods for person i 
		grad = function(par, X) {
		
			eta = as.matrix(dat$EVT)
			offs = dat$OFFS
	
			X_par = X %*% par	
			exp_X_par = exp(X_par)
			offs_exp_X_par = Matrix(offs * exp_X_par)

			# denominator term for "pi" vector
				denom_pid = t(pid_mat) %*% offs_exp_X_par
			
			scale_pid = pid_mat %*% (n_pid/denom_pid)

			# holds the (n_i * pi_ik) terms
			scale_pi_pid = scale_pid * offs_exp_X_par
	
			gradient = as.numeric(t(y - scale_pi_pid) %*% X)
	
			return(gradient)
		}


	# fisher information where X is the sparse matrix of covariates (drugs), 
	# representing a matrix of dimension (total numpds x num drugs)
		fisherinfo <- function(par, X) {
		
			p = dim(X)[2]
	
			X_par = X %*% par	
			exp_X_par = exp(X_par)
			offs_exp_X_par = Matrix(offs * exp_X_par)

			# denominator term
			denom_pid = t(pid_mat) %*% offs_exp_X_par
					
			# numerator term			
			x_offs_exp_X_par = X * as.numeric(offs_exp_X_par)
			numer_pid_mat = t(pid_mat) %*% x_offs_exp_X_par
				
			# quotient in first term	
			quot_pid_mat = numer_pid_mat / as.numeric(denom_pid)
			term1 = t(apply(quot_pid, 1, function(x) { x %o% x }))
				
			# quotient in second term
			X_outer = Matrix(t(apply(X, 1, function(x) { x %o% x })))
		
			pid_mat_offs_exp_X_par = pid_mat * as.numeric(offs_exp_X_par)
			term2 = (t(pid_mat_offs_exp_X_par) %*% X_outer) / as.numeric(denom_pid)

			total = Matrix(-as.numeric(t(n_pid) %*% (term1 - term2)), nrow=p, ncol=p)

			return(total)
		}


	# BBR type update for parameter component beta_j
	update_beta_j <- function(j, beta, X, prior) {
	
			X_beta = as.numeric(X %*% beta)
			exp_X_beta = exp(X_beta)
			offs_exp_X_beta = offs * exp_X_beta
			x_offs_exp_X_beta = X[,j] * offs_exp_X_beta
		
		# denominator term
		denom_pid = apply(PIDlist, 1, function(pid) { 
			sum(offs_exp_X_beta[dat$PID == pid]) 
		})
			
		# numerator term
		numer_pid = apply(PIDlist, 1, function(pid) { 
			sum(x_offs_exp_X_beta[dat$PID == pid]) 
		})
			
		# scale by denominator
		t1 = numer_pid / denom_pid

		if (prior == "normal") {		# normal prior
			
			g_d1 = (n_pid %*% t1) - (X[,j] %*% eta) + (beta[j]/sigma2_beta)
			g_d2 = (n_pid %*% (t1 * (1-t1))) + (1/sigma2_beta)

			update = -as.numeric(g_d1/g_d2)

		} else {								# laplace prior
			
			g_d2 = n_pid %*% (t1 * (1-t1))
			g_d1_first = (n_pid %*% t1) - (X[,j] %*% eta)

			if ( sign(beta[j]) == 0 ) {
				neg_update = -as.numeric((g_d1_first - lambda)/g_d2)
				pos_update = -as.numeric((g_d1_first + lambda)/g_d2)
				
				# only one of these conditions will hold, from convexity
					update = ifelse( neg_update < 0, neg_update,
							ifelse( pos_update > 0, pos_update, 0) )
							
			} else {
				update = -as.numeric((g_d1_first + lambda*sign(beta[j]))/g_d2)
				
				# update would change sign, so return beta_j to 0
				if (sign(beta[j] + update) != sign(beta[j])) {
					update = -beta[j]
				}
			}
		}
		
		return( update )
	}
	
	
	
# RUN -------------------------------------------------------------------------
	# read in data
	#datafile = "/Users/ses/Desktop/sccs_multivar_package/ge_bbrtest/cond_373474_ge_bbrtest_Rin.txt"
	#datafile = "data/sm_ge_format_multivar_OUT_53drugs.txt"

	# data in format of flat tab-delimited text file, header: 
	#	PID	EVT	OFFS	D1	D2	D3	... etc.
	dat = read.table(datafile, fill=TRUE, row.names=NULL, 
				header=TRUE, sep="\t")        # will automatically pad with NAs
												        # for rows w/o max num of drugs
	nrows = dim(dat)[1]
	ncols = dim(dat)[2]
	drugcols = paste("D", 1:(ncols-3), sep="")

	dat$PID = as.numeric(dat$PID)
		
	# vector containing all (unique) drug id numbers
		druglist = unique(as.numeric(as.matrix(dat[drugcols])))
		druglist = druglist[!is.na(druglist)]
		druglist = druglist[druglist != 0]
		druglist = sort(druglist)
		ndrugs = length(druglist)

		# unique PIDS
		PIDlist = as.matrix(unique(dat$PID))

		# sparseMatrix: covariate matrix, in sparse format (in "Matrix" package)
			suppressPackageStartupMessages(
				require(Matrix, warn.conflicts=FALSE, quietly=TRUE)
			)
			
			indices = unlist(apply(dat[,drugcols], 2, function(col) {
				rbind(
					which(!is.na(col)),
					pmatch(col[!is.na(col)], druglist, duplicates.ok=TRUE)
					)}), use.names=FALSE)
			
			indices = array(indices, dim=c(2, length(indices)/2))
			
			indices = indices[,which(!is.na(indices[2,]))]
			
			X.sM = sparseMatrix(i=indices[1,], j=indices[2,], dims=c(nrows, ndrugs))

			X.sM = as(X.sM, "dgCMatrix")
	
	y = Matrix(dat$EVT)
	offs = dat$OFFS
		
		
	# blockwise matrix to keep track of PIDs
	pid_mat = sparseMatrix(i=1:nrows, j=pmatch(dat$PID, PIDlist, duplicates.ok=TRUE), 
				dims=c(nrows, length(PIDlist)))
				
	pid_mat = as(pid_mat, "dgCMatrix")
				
	# number of events for each person
	n_pid = t(pid_mat) %*% y
				

# loop to do BBR-type procedure ---------------------------------------------------

	# set prior parameters
		#sigma2_beta = 10
		#lambda = sqrt(2/20)

		sigma2_beta = prior_var
		lambda = sqrt(2/prior_var)
	
	# other initializations
	#	type = "laplace"
		done = FALSE
		beta = numeric(ndrugs)
		delta = rep(2, ndrugs)
		iter = 1

		X_beta = as.numeric(X.sM %*% beta)
		e_Xbeta = exp(X_beta)
		offs_e_Xbeta = Matrix(offs * e_Xbeta)
		
system.time(
		while(!done) {
			beta_old = beta
		
			for (j in 1:ndrugs) {
			
				# newton-raphson step for coordinate j
				x_j = Matrix(X.sM[,j])
				
				x_offs_e_Xbeta = x_j * offs_e_Xbeta

				numer_j = t(pid_mat) %*% x_offs_e_Xbeta
				denom = t(pid_mat) %*% offs_e_Xbeta
			
				quot_j = numer_j / denom
				
				grad_j = (t(n_pid) %*% quot_j) - (t(y) %*% x_j)
				hess_j = (t(n_pid) %*% (quot_j * (1-quot_j)))
				
				if (prior == "normal") { # NORMAL
					
					grad_j = grad_j + (beta[j]/sigma2_beta)
					hess_j = hess_j + (1/sigma2_beta)
			
					del_beta = as.numeric(-grad_j / hess_j)
					
				} else {	 # LAPLACIAN	

					if (beta[j] == 0) {
						neg_update = -as.numeric((grad_j - lambda)/hess_j)
						pos_update = -as.numeric((grad_j + lambda)/hess_j)
					
						# only one of these conditions will hold, from convexity
						del_beta = ifelse( neg_update < 0, neg_update,
								ifelse( pos_update > 0, pos_update, 0) )
					} else {
						
						del_beta = -as.numeric((grad_j + lambda*sign(beta[j]))/hess_j)
				
						# update would change sign, so return beta_j to 0
						if (sign(beta[j] + del_beta) != sign(beta[j])) {
							del_beta = -beta[j]
						}
					}
				}
				
				# limit update to trust region
				del_beta = ifelse(del_beta < -delta[j], -delta[j], 
							ifelse(del_beta > delta[j], delta[j], del_beta))
			
				delta[j] = max(2*abs(del_beta), 0.5*delta[j])
				beta[j] = beta[j] + del_beta

				
				# update vector offs * exp( X * beta)
				del_beta_xj = del_beta * x_j
				X_beta = X_beta + del_beta_xj
				offs_e_Xbeta = offs_e_Xbeta * exp(del_beta_xj)
			}

			# update log posterior
			loglik = (t(y) %*% X_beta) - (t(n_pid) %*% log(t(pid_mat) %*% offs_e_Xbeta))
				
			if (prior == "normal") {
				logpost = loglik - 0.5*ndrugs*log(2*pi*sigma2_beta) - 0.5*(t(beta) %*% beta)/sigma2_beta
			} else {
				logpost = loglik + ndrugs*log(0.5*lambda) - lambda*sum(abs(beta))
			}
				
			logpost = as.numeric(logpost)
			
			
			cat(beta, "\n")
			cat("logpost: ", logpost, "(iter:",iter,")\n\n")
			iter = iter + 1
		
				# calculate and check convergence criteria
					conv = as.numeric(
						abs(
							t(X.sM %*% (beta - beta_old)) %*%y
						)
					)
					conv = conv / as.numeric(1 + 
						abs(
							t(X.sM %*% beta) %*% y
						)
					)
		
					if (as.numeric(conv	) <= 0.0001) { done = TRUE }
		}
)	# end system.time

	
# ---------------------- print out results ----------------------
	for (i in 1:ndrugs) {
		cat(druglist[i], "\t", beta[i], "\n", sep="")
	}


# ML procedures
#	library(maxLik)
#	ML = maxLik(logLik=function(par) { l(par, X.sM) }, grad=function(par) { grad(par, X.sM) },
#		hess = function(par) { -fisherinfo(par, X.sM) }, start=numeric(ndrugs), reltol=0.0001)
	

