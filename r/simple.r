library(randomForest)
library(miscTools)
library(ggplot2)

df <- read.csv("wine.csv")

df$is_red <- factor(ifelse(df$color=='red', 1, 0))
df$high_quality <- factor(ifelse(df$quality > 6, 1, 0))
df$quality <- factor(df$quality)

cols <- names(train)[1:12]
system.time(clf <- randomForest(factor(quality) ~ ., data=train[,cols], ntree=20, nodesize=5, mtry=9))
# user system elapsed
# 0.366 0.006 0.372
 
table(test$quality, predict(clf, test[cols]))
# 3 4 5 6 7 8 9
# 3 0 0 4 5 0 0 0
# 4 0 0 35 22 1 0 0
# 5 0 0 375 137 6 1 0
# 6 0 1 129 531 52 1 0
# 7 0 0 13 113 150 2 0
# 8 0 0 0 19 13 18 0
# 9 0 0 0 3 0 0 0
sum(test$quality==predict(clf, test[cols])) / nrow(test)

cols <- c('is_red', 'fixed.acidity', 'density', 'pH', 'alcohol')
rf <- randomForest(alcohol ~ ., data=train[,cols], ntree=20)

(r2 <- rSquared(test$alcohol, test$alcohol - predict(rf, test[,cols])))
(mse <- mean((test$alcohol - predict(rf, test[,cols]))^2))

p <- ggplot(aes(x=actual, y=pred),
	    data=data.frame(actual=test$alcohol, pred=predict(rf, test[,cols])))

p + geom_point() + geom_abline(color="red") + ggtitle(paste("Random Forest Regression in R r^2=", r2, sep=""))
