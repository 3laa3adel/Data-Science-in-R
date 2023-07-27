#Loading required packages
library(forcats);library(corrplot);library(caret);library(tidyverse);library(varrank)
library(randomForest);library(stringr);library(glmnet);library(fastDummies);library(mlr)
library(PerformanceAnalytics);library(Metrics);library(lightgbm);library(DataExplorer)
library(recipes);library(dplyr);library(mltools);library(data.table);library(gridExtra);library(kernlab)
library(rpart);library(rpart.plot);library(rattle);library(RColorBrewer);library(e1071);library(class)
# clean everything done before
rm(list=ls())
# read training and testing data set
train <- read.csv("./train.csv")
test <- read.csv("./test.csv")
#########################################################################################################
#EDA & Visualizations
str(train)
plot_missing(train)
tbl <- with(train, table(Sex))
ggplot(as.data.frame(tbl), aes(factor(Sex), Freq, fill = Sex)) +     
  geom_col(position = 'dodge')
tbl <- with(train, table(Survived))
ggplot(as.data.frame(tbl), aes(factor(Survived), Freq, fill = Survived)) +     
  geom_col(position = 'dodge')
############################################################################
str(test)
plot_missing(test)
tbl <- with(test, table(Sex))
ggplot(as.data.frame(tbl), aes(factor(Sex), Freq, fill = Sex)) +     
  geom_col(position = 'dodge')
#########################################################################################################
#Preparing Data


# Change Sex to 0 = male, 1 = female
train$Sex <- sapply(as.character(train$Sex), switch, 'male' = 0, 'female' = 1)
test$Sex <- sapply(as.character(test$Sex), switch, 'male' = 0, 'female' = 1)

#Removing NA Values
train_age <- na.omit(train$Age)
train_age_avg <- mean(train_age)
train$Age[is.na(train$Age)] <- train_age_avg

test_age <- na.omit(test$Age)
test_age_avg <- mean(test_age)
test$Age[is.na(test$Age)] <- test_age_avg

test_fare <- na.omit(test$Fare)
test_fare_avg <- mean(test_fare)
test$Fare[is.na(test$Fare)] <- test_fare_avg

# Change Age to 0 = Adult(>=18), 1 = Child(<18)
train$Age <- ifelse(train$Age<18, 1, 0)
test$Age <- ifelse(test$Age<18, 1, 0)
#Function to normalize the values
normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

#Call Function to normalize values
train$Pclass = normalize(train$Pclass)
test$Pclass = normalize(test$Pclass)

test_length <- length(test$Fare)
fare <- normalize(c(train$Fare, test$Fare))
train$Fare <- fare[1:(length(fare)-test_length)]
test$Fare <- fare[(length(fare)-test_length + 1): length(fare)]

survived <- train$Survived
passengers <- test$PassengerId
train$Survived<-NULL
train$PassengerId<-NULL
test$PassengerId<-NULL
train<-train%>%select(-c("Ticket", "Cabin", "Name","take.off"))
test<-test%>%select(-c("Ticket", "Cabin", "Name","take.off"))
sum(is.na(train))
sum(is.na(test))
#######################################################################################
#model Random Forest(Best Model)
######################################################################################
rf_model <- randomForest(
  as.factor(survived)~., 
  data=train)

rf_model
prediction <- predict(rf_model, train)
#######################################################################################
#create confusion matrix
cfMatrix_rf <- confusionMatrix(
  data = relevel(prediction, ref = "1"),
  reference = relevel(as.factor(survived), ref = "1")
)
cfMatrix_rf
#Apply the model to test data
rf_prediction <- predict(rf_model, test)
df_final<-data.frame(PassengerId =passengers,Survived=rf_prediction )
write.csv(df_final, "rf_result.csv", row.names =F, quote=F)
head(df_final)
#When Upload rf_result in kaggle get 77.7% Accuracy(score)
################################################################################################################
######################################################################################
nv_model <- naiveBayes(
  as.factor(survived)~., 
  data=train)

nv_model
prediction <- predict(nv_model, train)
caret::confusionMatrix(prediction,as.factor(survived))
#######################################################################################
#Apply the model to test data
nv_prediction <- predict(nv_model, test)
df_final<-data.frame(PassengerId =passengers,Survived=nv_prediction )
write.csv(df_final, "nv_result.csv", row.names =F, quote=F)
head(df_final)
#When Upload rf_result in kaggle get 74% Accuracy(score)
################################################################################################################
#Decision Tree Model
dt_model <- rpart(
  as.factor(survived)~., 
  data=train)

dt_model
fancyRpartPlot(dt_model)

prediction <- predict(dt_model, train,type='class')
###############################################################################################################
#create confusion matrix
cfMatrix_dt <- caret::confusionMatrix(prediction,as.factor(survived))
cfMatrix_dt
#Apply the model to test data
dt_prediction <- predict(dt_model, test,type='class')
df_final<-data.frame(PassengerId =passengers,Survived=dt_prediction )
write.csv(df_final, "rf_result.csv", row.names =F, quote=F)
head(df_final)
#When Upload DT_result in kaggle get 77.9% Accuracy(score)
##############################################################################################################

