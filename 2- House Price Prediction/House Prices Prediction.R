library(forcats);library(corrplot);library(caret);library(tidyverse);library(varrank)
library(randomForest);library(stringr);library(glmnet);library(fastDummies);library(mlr)
library(PerformanceAnalytics);library(Metrics);library(lightgbm);library(DataExplorer)
library(recipes);library(dplyr);library(mltools);library(data.table);library(gridExtra);library(kernlab)
# clean everything done before
rm(list=ls())
# read training and testing data set
train <- read.csv2("./train.csv", sep=",", stringsAsFactors = TRUE)
test <- read.csv2("./test.csv", sep=",", stringsAsFactors = TRUE)
######################################################################
#Visualizations and EDA
######################################################################
# histograms of integer variables
train %>%
  keep(is.numeric) %>%   
  gather() %>%                  
  ggplot(aes(value)) + 
  facet_wrap(~ key, scales = "free") +
  geom_histogram()
######################################################################
p<-plot_missing(train)
######################################################################
par(mfrow=c(1,2))
hist(train$SalePrice)
boxplot(train$SalePrice)
######################################################################
hist(train$SalePrice, freq=F)
s<-seq(0,770000,100)
lines(s,dlnorm(s,12.024,0.4,log=F), col="blue")
######################################################################
num_index<-map(train,class) %in% c("integer","numeric") # index for numerical columns
cat_index<-!num_index # index for categorical columns
house_cat<-train[,cat_index] %>% mutate(SalePrice=train$SalePrice)
house_num<-train[,num_index] 
corrplot(cor(house_num), method="circle")
######################################################################
boxplot(train$MasVnrArea~train$MasVnrType)
abline(h=198, col="red")
######################################################################
#Preprocessing
# store and remove variable Id
trainId <- train$Id
testId <- test$Id
train$Id <- NULL
test$Id <- NULL
trainSalePrice <- train$SalePrice
train$SalePrice <- NULL
cbind(c("Training", "Testing"),
      rbind(dim(train), dim(test)))
# convert factor into dummy variables
dummies <- dummyVars(train, data = train)

# apply transformation to the training data set
trainDummy <- as.data.frame(predict(dummies, newdata = train))

# view dimensions
cbind(c("Training with Factor Variables","Training with Dummy Variables"),
      rbind(dim(train), dim(trainDummy)))
# apply transformation on the training data set to the testing data set
testDummy <- as.data.frame(predict(dummies, newdata = test))

# view dimensions
cbind(c("Testing with Factor Variables","Testing with Dummy Variables"),
      rbind(dim(test), dim(testDummy)))

# change column names from the training data set
names(trainDummy)[names(trainDummy) == "MSZoning.C (all)"] <- "MSZoning.C"
names(trainDummy)[names(trainDummy) == "Exterior1st.Wd Sdng"] <- "Exterior1st.WdSdng"
names(trainDummy)[names(trainDummy) == "Exterior2nd.Wd Sdng"] <- "Exterior2nd.WdSdng"
names(trainDummy)[names(trainDummy) == "Exterior2nd.Brk Cmn"] <- "Exterior2nd.BrkComm"
names(trainDummy)[names(trainDummy) == "RoofMatl.Tar&Grv"] <- "RoofMatl.Tar.Grv"
names(trainDummy)[names(trainDummy) == "Exterior2nd.Wd Shng"] <- "Exterior2nd.WdShing"

# change column names from the testing data set
names(testDummy)[names(testDummy) == "MSZoning.C (all)"] <- "MSZoning.C"
names(testDummy)[names(testDummy) == "Exterior1st.Wd Sdng"] <- "Exterior1st.WdSdng"
names(testDummy)[names(testDummy) == "Exterior2nd.Wd Sdng"] <- "Exterior2nd.WdSdng"
names(testDummy)[names(testDummy) == "Exterior2nd.Brk Cmn"] <- "Exterior2nd.BrkComm"
names(testDummy)[names(testDummy) == "RoofMatl.Tar&Grv"] <- "RoofMatl.Tar.Grv"
names(testDummy)[names(testDummy) == "Exterior2nd.Wd Shng"] <- "Exterior2nd.WdShing"

null <- colMeans(is.na(trainDummy)) * 100
#remove columns with null vals 45 perc 
trainNoHighNA <- trainDummy[, null<=45]


# view dimensions
cbind(c("Training with High NA Variables","Training without High NA Variables"),
      rbind(dim(trainDummy), dim(trainNoHighNA)))

null <- colMeans(is.na(testDummy)) * 100
#remove columns with null vals 45 perc 
testNoHighNA <- testDummy[, null <= 45]
# view dimensions
cbind(c("Testing with High NA Variables","Testing without High NA Variables"),
      rbind(dim(testDummy), dim(testNoHighNA)))

# run imputation
#Imputing missing values
#Replace the column's missing value with zero.
#Replace the column's missing value with the mean.
#Replace the column's missing value with the median.
preProcImpute <- preProcess(trainNoHighNA, method="bagImpute")

# apply imputation to the training data set
trainImpute <- predict(preProcImpute, trainNoHighNA)

# apply imputation to the testing data set
testImpute <- predict(preProcImpute, testNoHighNA)

# remove near zero variance
nzv <- nearZeroVar(trainImpute)
trainNoNzv <- trainImpute[,-nzv]
testNoNzv <- testImpute[,-nzv]

# create a correlation matrix
trainMatrixCor <-  cor(trainNoNzv) 

# find variables highly correlated
trainHighlyCor <- findCorrelation(trainMatrixCor, cutoff = .75) 

# remove variables highly correlated
trainLowCor <- trainNoNzv[,-trainHighlyCor]

# training dimensions
cbind(c("Training with High Correlation","Training without High Correlation"),
      rbind(dim(trainNoNzv), dim(trainLowCor)))
# apply removal to the testing data set
testLowCor <- testNoNzv[,-trainHighlyCor]

# testing dimensions
cbind(c("Testing with High Correlation","Testing without High Correlation"),
      rbind(dim(testNoNzv), dim(testLowCor)))

# run other transformation
preProcNorm <- preProcess(trainLowCor, method=c("YeoJohnson","spatialSign"))

# apply imputation to the training data set
trainT <- predict(preProcNorm, trainLowCor)

# apply imputation to the testing data set
testT <- predict(preProcNorm, testLowCor)
######################################################################
#Modeling
######################################################################
#Linear Regression
lm_mod <- lm(trainSalePrice~., data = trainT)
paste("Linear Regression MSE: ",mean(lm_mod$residuals^2))
trainPredLM <- predict(lm_mod, trainT)
plot(trainPredLM,                                # Draw plot using Base R
     trainSalePrice,
     xlab = "Predicted Values",
     ylab = "Observed Values")
abline(a = 0,                                        # Add straight line
       b = 1,
       col = "red",
       lwd = 2)

# get predicted values on the testing data set
lm_pred <- predict(lm_mod, testT)
# build submission file
submission <- as.data.frame(cbind(testId, lm_pred))

# change column names
colnames(submission)[1] <- "Id"
colnames(submission)[2] <- "SalePrice"

# exponentiate predicted values
submission$SalePrice <- submission$SalePrice
head(submission)
# export file
write.csv(submission,"./LinearRegrission.csv", row.names = FALSE)
######################################################################
#Logistic Regression
lo_mod <- glm(trainSalePrice~., data = trainT)
paste("Logistic Regression MSE: ",mean(lo_mod$residuals^2))
trainPredLO <- predict(lo_mod, trainT)
plot(trainPredLO,                                # Draw plot using Base R
     trainSalePrice,
     xlab = "Predicted Values",
     ylab = "Observed Values")
abline(a = 0,                                        # Add straight line
       b = 1,
       col = "red",
       lwd = 2)


# get predicted values on the testing data set
lO_pred <- predict(lo_mod, testT)
# build submission file
submission <- as.data.frame(cbind(testId, lO_pred))

# change column names
colnames(submission)[1] <- "Id"
colnames(submission)[2] <- "SalePrice"

# exponentiate predicted values
submission$SalePrice <- submission$SalePrice
head(submission)
# export file
write.csv(submission,"./LogisticRegrission.csv", row.names = FALSE)
######################################################################
RF_mod<-randomForest(x = trainT,
                     y = trainSalePrice,ntree=500)
paste("Random Forest MSE: ",mean((predict(RF_mod,trainT)-trainSalePrice)^2))
trainPredRF <- predict(RF_mod, trainT)
plot(trainPredRF,                                # Draw plot using Base R
     trainSalePrice,
     xlab = "Predicted Values",
     ylab = "Observed Values")
abline(a = 0,                                        # Add straight line
       b = 1,
       col = "red",
       lwd = 2)


# get predicted values on the testing data set
RF_pred <- predict(RF_mod, testT)
# build submission file
submission <- as.data.frame(cbind(testId, RF_pred))

# change column names
colnames(submission)[1] <- "Id"
colnames(submission)[2] <- "SalePrice"

# exponentiate predicted values
submission$SalePrice <- submission$SalePrice
head(submission)
# export file
write.csv(submission,"./RandomForest.csv", row.names = FALSE)
######################################################################

#SVM
# tuning parameters
set.seed(231)
sigDist <- sigest(trainSalePrice ~ ., data = trainT, frac = 1)
svmTuneGrid <- data.frame(sigma = as.vector(sigDist)[1], C = 2^(-2:7))

# fit support vector machine (SVM) model
set.seed(1056)
modSVM <- caret::train(x = trainT,
                y = trainSalePrice,
                method = "svmRadial",
                tuneGrid = svmTuneGrid,
                trControl = trainControl(method = "boot", number = 50))
modSVM
paste("SVM MSE: ",mean((predict(modSVM, trainT)-trainSalePrice)^2))
######################################################################
trainPredSVM <- predict(modSVM, trainT)
plot(trainPredSVM,                                # Draw plot using Base R
     trainSalePrice,
     xlab = "Predicted Values",
     ylab = "Observed Values")
abline(a = 0,                                        # Add straight line
       b = 1,
       col = "red",
       lwd = 2)


# get predicted values on the testing data set
testPredSVM <- predict(modSVM, testT)

# build submission file
submission <- as.data.frame(cbind(testId, testPredSVM))

# change column names
colnames(submission)[1] <- "Id"
colnames(submission)[2] <- "SalePrice"

# exponentiate predicted values
submission$SalePrice <- submission$SalePrice
head(submission)
# export file
write.csv(submission,"./SVM.csv", row.names = FALSE)
#when upload submission.csv to kaggle get score 0.13
################################################################################
models <- c("SVM", "Random Forest", "Linear Regression", "Logistic Regression")
mse <- c(563470860.62442, 161017101.542508, 1052446698.04573, 1052446698.04573)
kaggle_scores <- c(0.13539, 0.15473, 0.17328, 0.17328)

data <- data.frame(models, mse, kaggle_scores)

lowest_mse_index <- which.min(data$mse)
lowest_kaggle_index <- which.min(data$kaggle_scores)

mse_colors <- rep("red", nrow(data))
mse_colors[lowest_mse_index] <- "green"
  
kaggle_colors <- rep("red", nrow(data))
kaggle_colors[lowest_kaggle_index] <- "green"
  # Bar plot for MSE
mse_plot <- ggplot(data, aes(x = models, y = mse)) +
  geom_bar(stat = "identity", fill = mse_colors) +
  labs(title = "Mean Squared Error (MSE) Comparison", x = "Models", y = "MSE") +
  theme_minimal() +
  theme(text = element_text(size = 14), plot.title = element_text(size = 18),
        axis.title = element_text(size = 16), axis.text = element_text(size = 18),
        axis.text.x = element_text(angle = 45, vjust = 0.5))

mse_plot <- mse_plot +
  geom_text(aes(label = sprintf("%.2f", mse), y = mse), vjust = -0.5, color = "black", size = 4)

print(mse_plot)
# Bar plot for Kaggle scores
kaggle_plot <- ggplot(data, aes(x = models, y = kaggle_scores)) +
  geom_bar(stat = "identity", fill = kaggle_colors) +
  labs(title = "Kaggle Scores Comparison", x = "Models", y = "Kaggle Score") +
  theme_minimal() +
  theme(text = element_text(size = 14), plot.title = element_text(size = 18),
        axis.title = element_text(size = 16), axis.text = element_text(size = 18),
        axis.text.x = element_text(angle = 45, vjust = 0.5))

kaggle_plot <- kaggle_plot +
  geom_text(aes(label = sprintf("%.5f", kaggle_scores), y = kaggle_scores), vjust = -0.5, color = "black", size = 4)

print(kaggle_plot)
