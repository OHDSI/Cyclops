# Using examples from Farrington's website http://statistics.open.ac.uk/sccs/r.htm,
# based on R scripts written by Heather Whitaker

getOxfordData <- function() {
    read.table(header = TRUE, text = "
indiv    eventday    start	end	exday
1	398	365	730	458
2	413	365	730	392
3	449	365	730	429
4	455	365	730	433
5	472	365	730	432
6	474	365	730	395
7	485	365	730	470
8	524	365	730	496
9	700	365	730	428
10	399	365	730	716
")  
}

constructOxfordDataFrame <- function() {
    dat <- getOxfordData()
    ex1 <- dat$exday + 14
    ex2 <- dat$exday + 35
    expo <- cbind(ex1, ex2)
    age <- c(547)
    expolev <- c(1, 0)
    
    ncuts <- ncol(expo) + length(age) + 2
    nevents <- nrow(dat)
    
    #create an ordered list of individual events and 
    #cut points for start, end, exposure and age groups
    ind <- rep(1:nevents, times = ncuts)
    cutp <- c(as.matrix(dat$start), as.matrix(dat$end), expo, rep(age, each = nevents))
    o <- order(ind, cutp)
    ind = as.factor(ind[o])
    cutp = cutp[o]
    
    #calculate interval lengths, set to 0 if before start or after end
    interval <- c(0, cutp[2:length(ind)]-cutp[1:length(ind)-1])
    interval <- ifelse(cutp<=dat$start[ind], 0, interval)
    interval <- ifelse(cutp>dat$end[ind], 0, interval)
    
    #event = 1 if event occurred in interval, otherwise 0
    event <- ifelse(dat$eventday[ind]>cutp-interval, 1, 0)
    event <- ifelse(dat$eventday[ind]<=cutp, event, 0)
    
    #age groups
    agegr <- cut(cutp, breaks = c(min(dat$start), age, max(dat$end)), labels = FALSE)
    agegr <- as.factor(agegr)
    
    #exposure groups
    exgr <- rep(0, nevents*ncuts)
    for(i in 1:ncol(expo)){
        exgr <- ifelse(cutp > expo[,i][ind], expolev[i], exgr)
    }
    exgr <- as.factor(exgr)
    
    #put all data in a data frame, take out data with 0 interval lengths
    data.frame(indiv = ind[interval!=0], 
               event = event[interval!=0], 
               interval = interval[interval!=0], 
               agegr = agegr[interval!=0], 
               exgr = exgr[interval!=0],              
               loginterval = log(interval[interval!=0]))        
}

oxford <- constructOxfordDataFrame()
