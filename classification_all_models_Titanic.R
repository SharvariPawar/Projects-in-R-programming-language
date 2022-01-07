#classification (binary): whether the passenger will survive or he will die
read.csv("titanic_train_clean_final.csv")->titanic
head(titanic)
titanic$X = NULL
sum(is.na(titanic))
colSums(titanic==-Inf)
View(titanic)
titanic$log_Fare = log(titanic$Fare + 1)
colSums(titanic==-Inf)

library(caTools)
set.seed(10)
split = sample.split(Y = titanic$Survived, SplitRatio = .8)
training = titanic[split, ]
test = titanic[!split, ]

#training the simple logistic regression model
glm(Survived~Fare, family = binomial, data = training)->model
summary(model)#aic 903.76
#lower the aic value better the model


#multiple logistic regression model
glm(Survived~., family = binomial, data = training)->model
summary(model) #aic 628.05

step(model, direction = "forward")->mod_forward
summary(mod_forward) #628.05


step(model, direction = "backward")->mod_backward
summary(mod_backward) #600.19


step(model, direction = "both")->mod_both
summary(mod_both) #600.19

#taking off deckE since showing less significant (p > 10%)
formula(mod_both)
f = 'Survived ~ Pclass + EmbarkedS + Family_Size + Titlemaster + Titlemiss + 
  log_Fare + Is_Male'
glm(formula = f, family = "binomial",data = training)-> mod2
summary(mod2) #600.61
#according to aic values mod_both is a better model
#but you are going to finalize using the cross-validations

library(car)
vif(mod2)
#to understand the impact of a less significant var deckE
#check vif values and handle multi-collinearity

predict(object = mod_both, newdata = test, type = "response")->p
ypred_test = ifelse(p>=.5,1,0)
ypred_test
length()

table(actual = test$Survived, predicted = ypred_test)->confusion_matrix
confusion_matrix
TN = confusion_matrix['0','0'] #correctly predicted as -ve
TP = confusion_matrix[2,2] #correct prediction as +ve
FP = confusion_matrix[1,2] #incorrectly predicted as +ve
FN = confusion_matrix[2,1] #incorrectly predicted as -ve

accuracy = (TP + TN)/(TP+TN + FP + FN)
accuracy

head(training)

factor_vars = c("Survived", "Pclass", "EmbarkedQ", "EmbarkedS", "Family_Size", "Age_State", "Titlelady", "Titlemajor", 
                "Titlemaster", "Titlemiss", "Titlemr", "Titlemrs", "Titlems", "Titleofficer", "DeckB", "DeckC", 
                "DeckD", "DeckE", "DeckF", "DeckG", "DeckN", "DeckT", "Is_Male", "Fare_Discretized")
titanic[,factor_vars]=as.data.frame(lapply(titanic[,factor_vars], as.factor))
str(titanic)
set.seed(10)
split = sample.split(Y = titanic$Survived, SplitRatio = .8)
training = titanic[split, ]
test = titanic[!split, ]

#Decision Trees
library(rpart)
model = rpart(as.factor(Survived)~., training)
library(rpart.plot)
prp(model)
model

#decision trees are prone to over-fitting 
#we'll prune the tree
plotcp(model)
mod = prune(tree = model, cp = .12)
prp(mod)

predict(object = model, newdata = test, type = "class")->ypred_test
table(actual = test$Survived, predicted = ypred_test)->confusion_matrix
confusion_matrix
TN = confusion_matrix['0','0'] #correctly predicted as -ve
TP = confusion_matrix[2,2] #correct prediction as +ve
FP = confusion_matrix[1,2] #incorrectly predicted as +ve
FN = confusion_matrix[2,1] #incorrectly predicted as -ve
accuracy = (TP + TN)/(TP+TN + FP + FN)
accuracy

#using unpruned model, the accuracy on test data was 78%

#Let's check the effect of pruning on accuracy
predict(object = mod, newdata = test, type = "class")->ypred_test
table(actual = test$Survived, predicted = ypred_test)->confusion_matrix
confusion_matrix
TN = confusion_matrix['0','0'] #correctly predicted as -ve
TP = confusion_matrix[2,2] #correct prediction as +ve
FP = confusion_matrix[1,2] #incorrectly predicted as +ve
FN = confusion_matrix[2,1] #incorrectly predicted as -ve
accuracy = (TP + TN)/(TP+TN + FP + FN)
accuracy

#by pruning the model, my test data accuracy has increased to 80%
#this shows that pruning is helping me in handling the variance error

#------------------------------------
#CLASS 10
#RANDOM FORESTS
library(randomForest)
set.seed(100)
model = randomForest(as.factor(Survived)~., data = training,ntree = 500, mtry = 3)
#mtry - as max features - sqrt(n_features) or log(n_features)

predict(model, newdata = test, type = "class")->ypred_test
table(actual = test$Survived, predicted = ypred_test)->confusion_matrix
confusion_matrix
TN = confusion_matrix['0','0'] #correctly predicted as -ve
TP = confusion_matrix[2,2] #correct prediction as +ve
FP = confusion_matrix[1,2] #incorrectly predicted as +ve
FN = confusion_matrix[2,1] #incorrectly predicted as -ve
accuracy = (TP + TN)/(TP+TN + FP + FN)
accuracy

#NAIVE BAYES 
#3 TYPES: GAUSSIAN NAIVE BAYES' - CONTINUOUS FEATURES ARE FOLLOWING THE 
#GAUSSIAN DISTRIBUTION
# BERNOULLI NAIVE BAYES - WHEN ALL THE FEATURES ARE BINARY
#MULTINOMIAL NAIVE BAYES - WHEN THE FEATURE ARE MULTINOMIAL (DISCRETE INTEGRAL VALUES)
#BERNOULLI AND MULTINOMIAL NAIVE BAYES ARE VERY COMMONLY USED FOR NLP PROBLEMS
#Naive Bayes only for classification models(not regression)

head(titanic)
which(names(training)=="Fare")->cn_Fare
training_nb = training[, -cn_Fare]
test_nb = test[, -cn_Fare]
head(training_nb)


#STANDARD SCALAR (STANDARD DEVIATION BASED SCALING)
scale(training_nb$log_Fare)->x
class(x)
dim(x)
x[,1]
library(dplyr)
training_nb = as_tibble(training_nb)
training_nb%>%select_if(is.numeric)%>%
  mutate(Age_Scaled = (Age - mean(Age))/sd(Age), 
         log_Fare_scaled = scale(log_Fare)[,1])%>%
  select(-c(Age, log_Fare))->scaled_cont_training


test_nb%>%select_if(is.numeric)%>%
  mutate(Age_Scaled = (Age - mean(Age))/sd(Age), 
         log_Fare_scaled = scale(log_Fare)[,1])%>%
  select(-c(Age, log_Fare))->scaled_cont_test
factor_vars
cbind(training_nb[, factor_vars], scaled_cont_training)->training_standardized
cbind(test_nb[, factor_vars], scaled_cont_test)->test_standardized
head(training_standardized)

#MIN MAX SCALAR (NORMALIZATION)
training_nb%>%select_if(is.numeric)%>%
  mutate(Age_Normalize = (Age - min(Age))/(max(Age) - min(Age)), 
      log_Fare_normalized = 
        (log_Fare - min(log_Fare))/(max(log_Fare) - min(log_Fare)))%>%
  select(-c(Age, log_Fare))->normalized_cont_training


test_nb%>%select_if(is.numeric)%>%
  mutate(Age_Normalize = (Age - min(Age))/(max(Age) - min(Age)), 
         log_Fare_normalized = 
           (log_Fare - min(log_Fare))/(max(log_Fare) - min(log_Fare)))%>%
  select(-c(Age, log_Fare))->normalized_cont_test

cbind(training_nb[, factor_vars], scaled_cont_training)->training_standardized
cbind(test_nb[, factor_vars], scaled_cont_test)->test_standardized
head(training_standardized)


cbind(training_nb[, factor_vars], normalized_cont_training)->training_normalized
cbind(test_nb[, factor_vars], normalized_cont_test)->test_normalized
head(training_normalized)


library(e1071)
model = naiveBayes(x = training_standardized[, -1], 
                   y = as.factor(training_standardized$Survived))
predict(model, newdata = test_standardized, type = "class")->ypred_test
table(actual = test_standardized$Survived, predicted = ypred_test)->confusion_matrix
confusion_matrix
TN = confusion_matrix['0','0'] #correctly predicted as -ve
TP = confusion_matrix[2,2] #correct prediction as +ve
FP = confusion_matrix[1,2] #incorrectly predicted as +ve
FN = confusion_matrix[2,1] #incorrectly predicted as -ve
accuracy = (TP + TN)/(TP+TN + FP + FN)
accuracy

#---------------------------------------------------------------
#KNN 
#based on k-nearest neighbors obtained using the distance calculations
#euclidean distance 
#Low value of k - results in low bias error / underfitting error
#But larger / medium large value of k leads to low variance error
#it's an adhoc algo, it's a lazy learner (doesn't learn until the test data arrives)
library(class)
head(training_standardized)
ypred_test = knn(train = 
        training_standardized[, c("Age_Scaled", "log_Fare_scaled")],
        test = test_standardized[,c("Age_Scaled", "log_Fare_scaled")],
        cl = as.factor(training_standardized$Survived), k = 5)
ypred_test
table(actual = test_standardized$Survived, predicted = ypred_test)->confusion_matrix
confusion_matrix
TN = confusion_matrix['0','0'] #correctly predicted as -ve
TP = confusion_matrix[2,2] #correct prediction as +ve
FP = confusion_matrix[1,2] #incorrectly predicted as +ve
FN = confusion_matrix[2,1] #incorrectly predicted as -ve
accuracy = (TP + TN)/(TP+TN + FP + FN)
accuracy
nrow(training)^.5 #i should check for the value of k atleast until 27
k_seq = seq(3, 71, 2)

acc = NULL
for(k in k_seq)
{
  ypred_test = knn(train = 
                     training_standardized[, c("Age_Scaled", "log_Fare_scaled")],
                   test = test_standardized[,c("Age_Scaled", "log_Fare_scaled")],
                   cl = as.factor(training$Survived), k = k)
  table(actual = test_standardized$Survived, predicted = ypred_test)->confusion_matrix
  TN = confusion_matrix['0','0'] #correctly predicted as -ve
  TP = confusion_matrix[2,2] #correct prediction as +ve
  FP = confusion_matrix[1,2] #incorrectly predicted as +ve
  FN = confusion_matrix[2,1] #incorrectly predicted as -ve
  accuracy = (TP + TN)/(TP+TN + FP + FN)
  acc = append(acc, accuracy)
}
acc
data.frame(k_seq, acc)->df
df$k_seq[df$acc ==max(df$acc)]
#k =35 is giving you the best value of accuracy
#you can also use cross-validation accuracy to find out the best value of k 
#by using a for loop for different values of k and finding out the average 
#5-fold cv score
#That value of k is selected where average cv score is maximum. Try out yourself as an assignment.

#SUPPORT VECTOR MACHINES 
#linear kernel
library(e1071)
model = svm(formula = as.factor(Survived)~., data = training_standardized,
            kernel = "linear")
predict(model, newdata = test_standardized, type = "class")->ypred_test
table(actual = test_standardized$Survived, predicted = ypred_test)->confusion_matrix
confusion_matrix
TN = confusion_matrix['0','0'] #correctly predicted as -ve
TP = confusion_matrix[2,2] #correct prediction as +ve
FP = confusion_matrix[1,2] #incorrectly predicted as +ve
FN = confusion_matrix[2,1] #incorrectly predicted as -ve
accuracy = (TP + TN)/(TP+TN + FP + FN)
accuracy

#gaussian kernel for the non-linear problems
library(e1071)
model = svm(formula = as.factor(Survived)~., data = training_standardized,
            kernel = "radial")
predict(model, newdata = test_standardized, type = "class")->ypred_test
table(actual = test_standardized$Survived, predicted = ypred_test)->confusion_matrix
confusion_matrix
TN = confusion_matrix['0','0'] #correctly predicted as -ve
TP = confusion_matrix[2,2] #correct prediction as +ve
FP = confusion_matrix[1,2] #incorrectly predicted as +ve
FN = confusion_matrix[2,1] #incorrectly predicted as -ve
accuracy = (TP + TN)/(TP+TN + FP + FN)
accuracy

#this is very commonly used to find out whether the data is linear or non-linear
#linear kernel will give a high accuracy for the linear data
#gaussing kernel will improve the accuracy substantially for a non-linear data



#Linear models for regression
#Linear Regressions and SVR Linear 

#Non linear models for regression?
#Decision trees, random forests, support vector regressions, knn regression

#Linear models for classification
#Logistic Regressions and SVM Linear 

#Non linear models for classification?
#Decision trees, random forests, support vector machines, knn, naive bayes

#find out the models that give you best accuracy on the test data
#Ultimately final selection of model you'll do using the 
#cross-validation score
#the model with highest average cross validation score and lowest std dev of cross validation is the best model


#Bias-Variance Tradeoff 
#Finalize the model using cross validations.
#train the model using entire data-set, i.e in the below code usint titanic_standardiz

library(e1071)
model = svm(formula = as.factor(Survived)~., data = training_standardized,
            kernel = "linear")

write.csv(test_standardized, file = "test_standardized.csv")
