# Title : Harvardx: 125.9x - Final Project.
# CYO Project Name: Credit Card Fraud Detection.
# Author: Vasu Gandhapudi (VG)
# Date: September 6, 2020

#******************************************************************#
#*      TITLE: CREDIT CARD FRAUD DETICITION
#******************************************************************#


#The aim of this R project is to build a classifier that can detect credit card fraudulent transactions. We will use a variety of machine learning algorithms that will be able to discern fraudulent from non-fraudulent one. By the end of this machine learning project, you will learn how to implement machine learning algorithms to perform classification.

##*********************************************************************************************************
# Import Library's for the project

if(!require(dplyr)) install.packages("dplyr", repos = "https://dplyr.tidyverse.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://ggplot2.tidyverse.org")
if(!require(pROC)) install.packages("pROC", repos = "https://web.expasy.org/pROC/")
if(!require(rpart)) install.packages("rPart", repos = "https://cran.r-project.org/package=rpart")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://www.milbo.org/rpart-plot")
if(!require(gbm)) install.packages("gbm", repos = "https://github.com/gbm-developers/gbm")
if(!require(randomForest)) install.packages("randomForest", repos = "https://www.stat.berkeley.edu/~breiman/RandomForests/")
if(!require(xgboost)) install.packages("xgboost", repos = "https://github.com/dmlc/xgboost")
if(!require(caTools)) install.packages("caTools", repos = "https://CRAN.R-project.org/package=caTools")
if(!require(ranger)) install.packages("ranger", repos = "https://github.com/imbs-hl/ranger")
##****************************************************************************************************************************

# Importing the dataset that contains transactions made by credit cards.
# average Best time to download: 1 to 2 min
creditcard_data <- fread("https://media.githubusercontent.com/media/vasu0907/Datasets/master/creditcard.csv", header = TRUE, sep = ",")

# Revive the concepts of R data frames -- Data Exploration
dim(creditcard_data)
head(creditcard_data)
tail(creditcard_data)
names(creditcard_data)
summary(creditcard_data)

# Standard deviation for the value name "Amount"
sd(creditcard_data$Amount)

# Data Manipulation

# verify that is there any NA values in the dataset
apply(creditcard_data,2,function(x) sum(is.na(x)))
creditcard_data %>% group_by(Class) %>% summarise(mean(Amount), median(Amount))

# we will scale our data using the scale() function. We will apply this to the amount component of our creditcard_data amount. 
# Scaling is also known as feature standardization. With the help of scaling, the data is structured according to a specified range. Therefore, there are no extreme values in our dataset that might interfere with the functioning of our model.
creditcard_data$Amount=scale(creditcard_data$Amount)
NewData=creditcard_data[,-c(1)]

# Data Modelling 
# Splitting the data set
set.seed(123)
data_sample <- sample.split(NewData$Class, SplitRatio = 0.80)
train_dataset <- subset(NewData, data_sample == TRUE)
test_dataset <- subset(NewData, data_sample == FALSE)
dim(train_dataset)
dim(test_dataset)

# Modelling technique for optimize  & Algorithms to predict. 

# Fitting Logistic Regression Model :1
#Logistic regression is a simple regression model whose output is a score between 0 and 1

Logistic_Model=glm(Class~., train_dataset, family = binomial())
summary(Logistic_Model)
# confusion Matrix
#confusion matrix is a very useful tool for calibrating the output of a model and examining all possible outcomes of your predictions (true positive, true negative, false positive, false negative).
# let us Use a threshold of 0.5 to transform predictions to binary

Confusion_matrix <- confusionMatrix(table(test_dataset$Class, as.numeric(predict(Logistic_Model, test_dataset, type = "response") > 0.5)))
print(Confusion_matrix)
# A simple logistic regression model achieved nearly 100% accuracy, with ~99% precision (positive predictive value) and ~100% recall (sensitivity). We can see there are only 7 false negatives (transactions which were fraudulent in reality but on identified as such by the model).
fourfoldplot(Confusion_matrix$table)
# In order to assess the perfomance of our model, we will describe the ROC curve.

# ROC is also known as Receiver Optimistic Characteristics.
lr.predict <- predict(Logistic_Model,train_dataset, probability = TRUE)
auc.gbm <- roc(train_dataset$Class, lr.predict)
plot(auc.gbm, main = paste0("AUC: ", round(pROC::auc(auc.gbm), 3)), col ="blue")

# ROC on unpredicted test data
lr.predict_roc <- predict(Logistic_Model,test_dataset,probability = TRUE)
auc.gbm <- roc(test_dataset$Class, lr.predict_roc)
plot(auc.gbm, main = paste0("AUC: ", round(pROC::auc(auc.gbm), 3)), col ="blue")

# Fitting a Decision Tree Models : 2
# It will take a little bit time for this decission model
# Decission Tree to plot the out come of decission, These outcomes are basically a consequence through which we can conclude as to what class the object belongs to. We will now implement our decision tree model and will plot it using the rpart.plot() function.
decissionTree_model <- rpart(Class~., creditcard_data, method = 'class')
predict_val <- predict(decissionTree_model, creditcard_data, type = 'class')
probability <- predict(decissionTree_model, creditcard_data, type = 'prob')
summary(decissionTree_model)

# Plot
rpart.plot(decissionTree_model)

# K-Fold with Cross-Validation: Model 3
# As we learnt: In terms of how to select  k  for cross validation, larger values of  k  are preferable but they will also take much more computational time. For this reason, the choices of  k=5  and  k=10  are common.
# we are using SMOTE resampling the data, with 5-fold cv and trains Random forest classifier with roc as a metric.
# K-fold cross Validation
CV_control <- trainControl(method = "cv",
                           number = 5,
                           verboseIter = T,
                           classProbs = T,
                           sampling = "smote",
                           summaryFunction = twoClassSummary,
                           savePredictions = T)

train_dataset_CV <- train_dataset
train_dataset_CV$Class <- as.factor(train_dataset_CV$Class)
levels(train_dataset_CV$Class) <- make.names(c(0,1))
model_CV <- train(Class~., data = train_dataset_CV, method = "rpart", trControl = CV_control, metric = 'ROC')
print(model_CV)
plot(model_CV)

# Now let see the model performing on unpredicted Test data.
Model_CV_predict <- predict(model_CV, test_dataset, type = "prob")
Confussion_matrix_CV <- confusionMatrix(table(as.numeric(Model_CV_predict$X1 > 0.5), test_dataset$Class))
print(Confussion_matrix_CV)

# ROC on unpredicted test data
model_CV_roc <- roc(test_dataset$Class, predict(model_CV, test_dataset, type = "prob")$X1)
print(model_CV_roc)
plot(model_CV_roc, main = paste0("AUC: ", round(pROC::auc(model_CV_roc), 3)), col = "blue")

# Method 2: Repeated k-fold Cross Validation

# The process of splitting the data into k-folds can be repeated a number of times, this is called Repeated k-fold Cross Validation. The final model accuracy is taken as the mean from the number of repeats.
CV_repeat <- trainControl(method = "repeatedcv",
                          number = 5,
                          repeats = 3,
                          verboseIter = T,
                          classProbs = T,
                          sampling = "smote",
                          summaryFunction = twoClassSummary,
                          savePredictions = T)
model_repet <- train(Class~., data = train_dataset_CV, trControl = CV_repeat, method = 'rpart2', metric = 'ROC')
print(model_repet)
plot(model_repet)

# Now let see the model performing on unpredicted Test data.

CV_repeat_pred <- predict(model_repet, test_dataset, type = 'prob')
confussion_Matrix_CVRepet <- confusionMatrix(table(as.numeric(CV_repeat_pred$X1 >0.5),test_dataset$Class))
print(confussion_Matrix_CVRepet)

# ROC on unpredicted test data

CV_repet_roc <- roc(test_dataset$Class, predict(model_repet, test_dataset, type = "prob")$X1)
print(CV_repet_roc)
plot(CV_repet_roc, main = paste0("AUC: ", round(pROC::auc(CV_repet_roc), 3)), col ="blue")

# Random forest Model: 4

Model_rf <- trainControl(method = "cv",
                         number = 5,
                         verboseIter = T,
                         classProbs = T,
                         sampling = "smote",
                         summaryFunction = twoClassSummary,
                         savePredictions = T)

train_dataset_rf <- train_dataset
train_dataset_rf$Class <- as.factor(train_dataset_rf$Class)
levels(train_dataset_rf$Class) <- make.names(c(0,1))
Model_rf <- train(Class~., data = train_dataset_rf, method = 'rf', trControl = Model_rf, metric = 'ROC')
print(Model_rf)
plot(Model_rf)
plot(varImp(Model_rf))

# Now let see the model performing on unpredicted Test data.

rf_pred <- predict(Model_rf, test_dataset, type = "prob")
Confusion_matrix_rf <- confusionMatrix(table(as.numeric(rf_pred$X1 > 0.5), test_dataset$Class))
print(Confusion_matrix_rf)

# Now we will look ROC on unpredicted test data

rf_roc <- roc(test_dataset$Class, predict(Model_rf, test_dataset, type = "prob")$X1)
print(rf_roc)
plot(rf_roc, main = paste0("AUC: ", round(pROC::auc(rf_roc), 4)), col ="blue")

# Model XG Boost: Gradient Boosted Tress: Model 5

train_XGB <- xgb.DMatrix(data = as.matrix(train_dataset[, -c("Class")]), label = as.numeric(train_dataset$Class)) 
Model_xgb <- xgboost(data = train_XGB, nrounds = 100, gamma = 0.1, max_depth = 10, objective = "binary:logistic", nthread = 7)
# we will look into unseen test Set
test_XBG <- xgb.DMatrix(data = as.matrix(test_dataset[, -c("Class")]), label = as.numeric(test_dataset$Class))
predit_xgb <- predict(Model_xgb,test_XBG)
confusionMatrix(table(as.numeric(predit_xgb > 0.5), test_dataset$Class))
#ROC
roc_xgb <- roc(test_dataset$Class, predit_xgb)
print(roc_xgb)
plot(roc_xgb, main = paste0("AUC: ", round(pROC::auc(roc_xgb), 4)), col = "blue")
