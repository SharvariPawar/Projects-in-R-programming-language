
#-------------------------------Titanic case study using caret package----------------------------------------------------------
#install.packages(c("e1071", "caret", "doSNOW", "ipred", "xgboost"))
library(caret)
library(doSNOW)

#=================================================================
# Load Data
#=================================================================

train <- read.csv("C:\\Users\\Vaibhav\\Desktop\\BA\\Datasets\\titanic_train.csv", stringsAsFactors = FALSE)
#View(train)




#=================================================================
# Data Wrangling
#=================================================================

# Replace missing embarked values with mode.
table(train$Embarked)
train$Embarked[train$Embarked == ""] <- "S"


#we find missing values in age 
#   It should b imputed based on multi-variate analysis. 
#   This can be done using caret package.
summary(train$Age)
train$MissingAge <- ifelse(is.na(train$Age),"Y", "N")
#An important information of family size is split across 2 cols - parents_children and sibling:
# Add a feature for family size (feature engineering)
train$FamilySize <- 1 + train$SibSp + train$Parch

# Set up factors.
train$Survived <- as.factor(train$Survived)
train$Pclass <- as.factor(train$Pclass)
train$Sex <- as.factor(train$Sex)
train$Embarked <- as.factor(train$Embarked)
train$MissingAge <- as.factor(train$MissingAge)


# Subset data to features we wish to keep/use.
features <- c("Survived", "Pclass", "Sex", "Age", "SibSp",
              "Parch", "Fare", "Embarked", "MissingAge",
              "FamilySize")
train <- train[, features]
str(train)


#=================================================================
# Impute Missing Values in Age Column
#=================================================================

# Caret supports a number of mechanism for imputing (i.e., 
# predicting) missing values. Leverage bagged decision trees
# to impute missing values for the Age feature.

#caret supports multiple methods of imputation like knn, bagged decision trees.
# bagged decision trees have the most predictive power but computation intensive 
# because it will create an imputation model on each of the cols. 
# This is a problem especially when we are processing data of 100 mn of rows. 

# The imputation methods in caret only work on numeric data. They don't work on factors. 

# First, transform all feature to dummy variables (means numeric variables using dummyVars fn in Caret package)
library(caret)
dummy.vars <- dummyVars(~ ., data = train[, -1])#excluding the dependent var in our model
View(dummy.vars)
train.dummy <- predict(dummy.vars, train[, -1])
View(train.dummy)
apply(X = train.dummy,2,FUN = class)#all cols are numeric now

# Now, impute!
library(ipred)
?preProcess #Pre-processing of predictors (centering, scaling etc.) can be 
#   estimated from the training data and applied to any dataset
#http://topepo.github.io/caret/pre-processing.html#imputation
colSums(is.na(train.dummy))
pre.process <- preProcess(train.dummy, method = c("bagImpute"))
#this will create an imputation model for every column of your data
imputed.data <- predict(pre.process, train.dummy) 
#this will predict the missing values according to various imputation methods 
#(here using bagging method)
View(imputed.data)
colSums(is.na(imputed.data))


train$Age <- imputed.data[, 6]



#=================================================================
# Split Data
#=================================================================

# Use caret to create a 70/30% split of the training data,
# keeping the proportions of the Survived class label the
# same across splits.
set.seed(54321)
indexes <- createDataPartition(train$Survived,
                               times = 1,
                               p = 0.7,
                               list = FALSE)
titanic.train <- train[indexes,]
titanic.test <- train[-indexes,]


# Examine the proportions of the Survived class lable across
# the datasets to validate if the sampling has resulted in similar proportions.
prop.table(table(train$Survived))
prop.table(table(titanic.train$Survived))
prop.table(table(titanic.test$Survived))


#=================================================================
# Train Model
#=================================================================

#Hyperparameter tuning using caret package
classifier = train(Survived ~ ., data = titanic.train, method = "xgbTree")
classifier
classifier$bestTune

# Make predictions on the test set using a xgboost model 
# trained on all 625 rows of the training set using the 
# found optimal hyperparameter values.
preds <- predict(classifier, titanic.test)


# Use caret's confusionMatrix() function to estimate the 
# effectiveness of this model on unseen, new data.
confusionMatrix(preds, titanic.test$Survived)

#=================================================================
# Train Model using grid-search for optimal hyperparameter values
#=================================================================

# Set up caret to perform 10-fold cross validation repeated 3 
# times and to use a grid search for optimal model hyperparamter
# values.

train.control <- trainControl(method = "cv",
                              number = 3,
                              search = "grid")
?trainControl
# process for the training the model is same for all the methods 
#   be random forest, xgboost, svm
#   in this case, we are specifying that we want to do repeated cv, 
#   3-fold cv repeated 3 times and we also want to do the grid search
#In the grid we'll define the hyper-parameters for the xgboost. 

getModelInfo("xgb")
modelLookup(model = "xgbTree")#so these are the parameters to be tuned


# Leverage a grid search of hyperparameters for xgboost. See 
# the following presentation for more information:
# https://www.slideshare.net/odsc/owen-zhangopen-sourcetoolsanddscompetitions1
tune.grid <- expand.grid(eta = c(0.05, 0.1),
                         nrounds = c(50, 100),
                         max_depth = c(6,7),
                         min_child_weight = 1,
                         colsample_bytree = 1,
                         gamma = 0,
                         subsample = 1)
View(tune.grid)#so this is testing on 243 different combinations to find the best model

#min_child_weight [default=1]

#Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results 
#in a leaf node with the sum of instance weight less than min_child_weight, 
#then the building process will give up further partitioning. 
#In linear regression task, this simply corresponds to minimum number of instances 
#needed to be in each node. The larger min_child_weight is, 
#the more conservative the algorithm will be.

#colsample_by_tree
#specifies the fraction of columns to be subsampled.

#subsample [default=1]
#Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost 
#would randomly sample half of the training data prior to growing trees. and this will prevent 
#overfitting. Subsampling will occur once in every boosting iteration.
#range: (0,1]

#Minimum loss reduction required to make a further partition on a leaf node of the tree. 
#The larger gamma is, the more conservative the algorithm will be.

caret.cv <- train(Survived ~ ., 
                  data = titanic.train,
                  method = "xgbTree",
                  tuneGrid = tune.grid,
                  trControl = train.control)

# Make predictions on the test set using a xgboost model 
# trained on all 625 rows of the training set using the 
# found optimal hyperparameter values.
preds <- predict(caret.cv, titanic.test)


# Use caret's confusionMatrix() function to estimate the 
# effectiveness of this model on unseen, new data.
confusionMatrix(preds, titanic.test$Survived)
#+ve class: 0
tp = 143
tn = 75
fp = 27
fn = 21
sensitivity = tp/(tp + fn)
specificity = tn/(tn+fp)
precision = tp/(tp + fp) #ppr
npr = tn/(tn+fn)

