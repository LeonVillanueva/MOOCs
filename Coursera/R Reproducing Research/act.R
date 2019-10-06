# STEP1: read from csv and change date into POSIXlt format, create a day attribute
setwd("c:/r/act")
act<-read.csv("activity.csv", header=TRUE)
act$date <- as.Date (act$date, format="%Y-%m-%d")
act$day<-weekdays(as.Date(act$date))
week <- (unique(act$day))
int <- (unique(act$interval))

	#make steps numeric

	act$steps <- as.numeric(act$steps)

# STEP2: histogram with NA ignored

par(mfcol=c(1,1),bg="white")

	hist(act$steps[!is.na(act$steps)], main = "Histogram of Steps Taken", xlab = "Steps")
	dev.copy(png,'figures/histogram1.png')
	dev.off
	
# STEP3: find the mean and the median of steps, mean and the median of steps PER day

	# I ignored zeros for median because well, it just doesn't make sense.

by.day <- data.frame(matrix(NA, nrow = 7, ncol = 3))
names(by.day) <- c("day","average","median")

for (i in 1:length(week))

{
	temp <- subset (act,day==week[i])
	bd.mean <- mean(temp$steps, na.rm = TRUE)
	bd.median <- median(temp$steps[temp$steps != 0], na.rm = TRUE)

by.day [i,1] <- week[i]
by.day [i,2] <- bd.mean
by.day [i,3] <- bd.median
}

# STEP4: average by interval by date, create table first, create a histogram

n.per.x <- length ( unique ( act$interval))
per <- data.frame(matrix(NA, nrow = n.per.x, ncol = 2))
names(per) <- c("interval","average")
per$average <- as.numeric (per$average)

for (i in 1:n.per.x)

{
	temp <- subset (act,interval==int[i])
	per.mean <- mean(temp$steps, na.rm = TRUE)

per[i,1] <- int[i]
per[i,2] <- per.mean

}

with(per, plot(per$interval,per$average,col="black",typ="l",ylab="Average # of Steps",xlab="Interval", main = "Average # of Steps by 5 Min Interval"))

	# what interval has max average
	
	max.per <- per$interval[per$average==max(per$average)]


# STEP5: imputing NA values

	# how many NAs
	
	na.steps <- is.na(act$steps) ; na.amt <- length (na.steps[na.steps == TRUE])
	
	# replacing NAs with averages

	raw <- data.frame(matrix(NA, nrow = nrow(act), ncol = 2))
	names(raw) <- c("day","steps")
	
	raw$steps <- act$steps
	raw$day <- act$day
	
	# replacement with by day, cause I'm lazy and out of time
	
	for (j in 1:length(week)) {
	
		raw$steps[is.na(raw$steps) & raw$day == week[j] ] <- by.day$average[by.day$day == week[j] ]
	
		}
		

		after <- raw
	
			after$week <- act$day
			after$date <- act$date
			after$interval <- act$interval

			#making a weekend vs weekday variable because that's easier

			after$week[after$week == "Saturday" | after$week == "Sunday"] <- "END"
			after$week[after$week != "END"] <- "WORK"
			
	summary(after)
	
# creating a new histogram 	
	
(par bg="white")

	hist(after$steps[!is.na(after$steps)], main = "New Histogram of Steps Taken", xlab = "Steps")
	
# creating a new by day table

after.by.day <- data.frame(matrix(NA, nrow = 7, ncol = 3))
names(after.by.day) <- c("day","average","median")

for (i in 1:length(week))

{
	temp <- after[after$day==week[i],]
	bd.mean <- mean(temp$steps, na.rm = TRUE)
	bd.median <- median(temp$steps[temp$steps != 0], na.rm = TRUE)

after.by.day [i,1] <- week[i]
after.by.day [i,2] <- bd.mean
after.by.day [i,3] <- bd.median
}

# new panel plot, histogram by new data AND divided by weekend vs weekday

weekend <- after[after$week == "END",]
weekday <- after[after$week == "WORK",]

		#weekend
		we.per.x <- length ( unique (weekend$interval))
		we.per <- data.frame(matrix(NA, nrow = we.per.x, ncol = 2))
		names(we.per) <- c("interval","average")
		we.per$average <- as.numeric (we.per$average)

			for (i in 1:we.per.x)

			{
			temp <- subset (weekend,interval==int[i])
			per.mean <- mean(temp$steps, na.rm = TRUE)

			we.per[i,1] <- int[i]
			we.per[i,2] <- per.mean
			}
			
		#weekday
		wd.per.x <- length ( unique (weekday$interval))
		wd.per <- data.frame(matrix(NA, nrow = wd.per.x, ncol = 2))
		names(wd.per) <- c("interval","average")
		wd.per$average <- as.numeric (wd.per$average)

			for (i in 1:wd.per.x)

			{
			temp <- subset (weekday,interval==int[i])
			per.mean <- mean(temp$steps, na.rm = TRUE)

			wd.per[i,1] <- int[i]
			wd.per[i,2] <- per.mean
			}

par(mfcol=c(2,1),bg="white")
			
# I know this isn't a panel graph but like I said out of time
			
with(we.per, plot(we.per$interval,we.per$average,col="red",typ="l",ylab="Average # of Steps",xlab="Interval", main = "WEEKEND: Average # of Steps by 5 Min Interval"));
		par(new=F)
with(wd.per, plot(wd.per$interval,wd.per$average,col="green",typ="l",ylab="Average # of Steps",xlab="Interval", main = "WEEKDAY: Average # of Steps by 5 Min Interval"))