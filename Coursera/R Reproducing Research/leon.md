# Reproducible Research: Peer Assessment 1

## Loading and preprocessing the data

Code to Process Data

```{r}
setwd("c:/r/act")
act<-read.csv("activity.csv", header=TRUE)
act$date <- as.Date (act$date, format="%Y-%m-%d")
act$day<-weekdays(as.Date(act$date))
week <- (unique(act$day))
int <- (unique(act$interval))
act$steps <- as.numeric(act$steps)
```

```{r}
head(act)
```

## What is mean total number of steps taken per day?

Code to get average steps per day

```{r}

```



## What is the average daily activity pattern?



## Imputing missing values



## Are there differences in activity patterns between weekdays and weekends?
