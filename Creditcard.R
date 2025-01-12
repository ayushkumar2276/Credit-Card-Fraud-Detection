# Required Libraries
library(pROC)
library(e1071)
library(dplyr)
library(factoextra)
library(ggplot2)
library(glmnet)
library(caret)
library(FNN)
library(kknn)

# Loading Train Data
trainData <- read.csv("train.csv")
trainData <- trainData[with(trainData, order(X)), ]

# Cleaning Amount Column
trainData$transactionAmount <- as.numeric(gsub('[$]', '', trainData$Amount))

# Extracting Hour from Time
trainData$transactionHour <- substring(trainData$Time, 1, 2)

# Creating Time Period Variables
trainData$isNight <- ifelse(trainData$transactionHour %in% c('00', '01', '02', '03', '04', '21', '22', '23'), 1, 0)
trainData$isMorning <- ifelse(trainData$transactionHour %in% c('08', '09', '10', '11', '12', '05', '06', '07'), 1, 0)
trainData$isAfternoon <- ifelse(trainData$transactionHour %in% c('16', '17', '18', '19', '20', '13', '14', '15'), 1, 0)

# Creating Fraud and Transaction Type Variables
trainData$isFraud <- ifelse(trainData$Is.Fraud. == "Yes", 1, 0)
trainData$isChip <- ifelse(trainData$Use.Chip == "Chip Transaction", 1, 0)
trainData$isSwipe <- ifelse(trainData$Use.Chip == "Swipe Transaction", 1, 0)
trainData$isOnline <- ifelse(trainData$Use.Chip == "Online Transaction", 1, 0)

# Categorizing Errors into Dummy Variables
trainData$isTechGlitch <- ifelse(trainData$Errors. %in% c(
  "Technical Glitch", "Bad PIN,Technical Glitch", "Insufficient Balance,Technical Glitch",
  "Bad Card Number,Technical Glitch", "Bad Zipcode,Technical Glitch"), 1, 0)
trainData$isBadCardNumber <- ifelse(trainData$Errors. %in% c("Bad Card Number", "Bad Card Number,Technical Glitch"), 1, 0)
trainData$isInsufficientBalance <- ifelse(trainData$Errors. %in% c(
  "Insufficient Balance", "Insufficient Balance,Technical Glitch", "Bad PIN,Insufficient Balance"), 1, 0)
trainData$isBadPin <- ifelse(trainData$Errors. %in% c(
  "Bad PIN", "Bad PIN,Technical Glitch", "Bad PIN,Insufficient Balance"), 1, 0)
trainData$isBadExpiration <- ifelse(trainData$Errors. %in% c("Bad Expiration", "Bad Expiration,Bad CVV"), 1, 0)
trainData$isBadCVV <- ifelse(trainData$Errors. %in% c("Bad CVV", "Bad Expiration,Bad CVV"), 1, 0)
trainData$isBadZipcode <- ifelse(trainData$Errors. %in% c("Bad Zipcode", "Bad Zipcode,Technical Glitch"), 1, 0)

# Removing Irrelevant Columns
trainData <- trainData[ , -1]
train <- select(trainData, -1:-6, -8:-16)
dfOriginal <- trainData

# Loading Test Data
testData <- read.csv("test.csv")
testData <- testData[with(testData, order(X)), ]

# Cleaning Amount Column
testData$transactionAmount <- as.numeric(gsub('[$]', '', testData$Amount))

# Extracting Hour from Time
testData$transactionHour <- substring(testData$Time, 1, 2)

# Creating Time Period Variables
testData$isNight <- ifelse(testData$transactionHour %in% c('00', '01', '02', '03', '04', '21', '22', '23'), 1, 0)
testData$isMorning <- ifelse(testData$transactionHour %in% c('08', '09', '10', '11', '12', '05', '06', '07'), 1, 0)
testData$isAfternoon <- ifelse(testData$transactionHour %in% c('16', '17', '18', '19', '20', '13', '14', '15'), 1, 0)

# Creating Fraud and Transaction Type Variables
testData$isFraud <- ifelse(testData$Is.Fraud. == "Yes", 1, 0)
testData$isChip <- ifelse(testData$Use.Chip == "Chip Transaction", 1, 0)
testData$isSwipe <- ifelse(testData$Use.Chip == "Swipe Transaction", 1, 0)
testData$isOnline <- ifelse(testData$Use.Chip == "Online Transaction", 1, 0)

# Categorizing Errors into Dummy Variables
testData$isTechGlitch <- ifelse(testData$Errors. %in% c(
  "Technical Glitch", "Bad PIN,Technical Glitch", "Insufficient Balance,Technical Glitch",
  "Bad Card Number,Technical Glitch", "Bad Zipcode,Technical Glitch"), 1, 0)
testData$isBadCardNumber <- ifelse(testData$Errors. %in% c("Bad Card Number", "Bad Card Number,Technical Glitch"), 1, 0)
testData$isInsufficientBalance <- ifelse(testData$Errors. %in% c(
  "Insufficient Balance", "Insufficient Balance,Technical Glitch", "Bad PIN,Insufficient Balance"), 1, 0)
testData$isBadPin <- ifelse(testData$Errors. %in% c(
  "Bad PIN", "Bad PIN,Technical Glitch", "Bad PIN,Insufficient Balance"), 1, 0)
testData$isBadExpiration <- ifelse(testData$Errors. %in% c("Bad Expiration", "Bad Expiration,Bad CVV"), 1, 0)
testData$isBadCVV <- ifelse(testData$Errors. %in% c("Bad CVV", "Bad Expiration,Bad CVV"), 1, 0)
testData$isBadZipcode <- ifelse(testData$Errors. %in% c("Bad Zipcode", "Bad Zipcode,Technical Glitch"), 1, 0)

# Removing Irrelevant Columns
testData <- testData[ , -1]
test <- select(testData, -1:-6, -8:-16)
# Train Logistic Regression Model
logisticModel <- glm(isFraud ~ ., data = train, family = binomial)

# Making Predictions for Training Data
train$predictedFraudProbability <- predict(logisticModel, type = "response")

# Evaluating Training Data Predictions
rocTrain <- roc(train$isFraud, train$predictedFraudProbability)
aucTrain <- auc(rocTrain)
cat("AUC for Training Data:", aucTrain, "\n")

# Making Predictions for Test Data
test$predictedFraudProbability <- predict(logisticModel, newdata = test, type = "response")

# Evaluating the Test Data Predictions
rocTest <- roc(test$isFraud, test$predictedFraudProbability)
aucTest <- auc(rocTest)
cat("AUC for Test Data:", aucTest, "\n")

# Visualizing ROC Curve for Test Data
rocTestPlot <- ggroc(rocTest) +
  ggtitle("ROC Curve for Test Data") +
  xlab("1 - Specificity") +
  ylab("Sensitivity") +
  theme_minimal()
print(rocTestPlot)

# Training K-Nearest Neighbors Model
knnModel <- train(
  isFraud ~ ., 
  data = train, 
  method = "knn", 
  tuneGrid = expand.grid(k = 1:10),
  trControl = trainControl(method = "cv", number = 5)
)

# Optimal K for KNN
optimalK <- knnModel$bestTune$k
cat("Optimal K for KNN:", optimalK, "\n")

# Predicting with KNN on Test Data
test$predictedFraudKnn <- knn(
  train = select(train, -isFraud), 
  test = select(test, -isFraud), 
  cl = train$isFraud, 
  k = optimalK
)

# Confusion Matrix for KNN Predictions
knnConfMatrix <- table(test$isFraud, test$predictedFraudKnn)
print(knnConfMatrix)

# Training SVM Model
svmModel <- svm(
  isFraud ~ ., 
  data = train, 
  kernel = "radial", 
  cost = 1, 
  gamma = 0.1
)

# Predicting with SVM on Test Data
test$predictedFraudSvm <- predict(svmModel, newdata = test)

# Confusion Matrix for SVM Predictions
svmConfMatrix <- table(test$isFraud, test$predictedFraudSvm)
print(svmConfMatrix)

# Training the Lasso Regression Model
lassoModel <- cv.glmnet(
  x = as.matrix(select(train, -isFraud)),
  y = train$isFraud,
  alpha = 1,
  family = "binomial"
)

# Optimal Lambda for Lasso
optimalLambda <- lassoModel$lambda.min
cat("Optimal Lambda for Lasso Regression:", optimalLambda, "\n")

# Prediction with Lasso on Test Data
test$predictedFraudLasso <- predict(
  lassoModel, 
  newx = as.matrix(select(test, -isFraud)), 
  s = optimalLambda, 
  type = "response"
)

# Lasso Predictions
rocLasso <- roc(test$isFraud, test$predictedFraudLasso)
aucLasso <- auc(rocLasso)
cat("AUC for Lasso Regression:", aucLasso, "\n")

# Summarizing the Results
summaryResults <- data.frame(
  Model = c("Logistic Regression", "KNN", "SVM", "Lasso Regression"),
  AUC = c(aucTest, NA, NA, aucLasso)
)

print(summaryResults)
