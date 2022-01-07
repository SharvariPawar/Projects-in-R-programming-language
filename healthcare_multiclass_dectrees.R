Claims = read.csv("C:\\Users\\Vaibhav\\Desktop\\BA\\Datasets\\ClaimsData.csv")
head(Claims)
#to predict the risk bucket of the customers in 2009
table(Claims$bucket2009)
table(Claims$bucket2008)

#baseline model - predicting that the person will not change the risk bucket next year
table(actual = Claims$bucket2009, predicted = Claims$bucket2008)->tab
#diag shows the customers who don't change the bucket in the next year
#risk bucket 2 > risk bucket 3 > risk  bucket 4 > risk bucket 5
#so a higher risk person going to a lower bucket will attract a higher penalty

baseline_accuracy = sum(diag(tab))/sum(tab)

#prediction is 2, where-as actually he belongs to risk bucket 1, penalty = 2-1
#prediction is 3, where-as actually he belongs to risk bucket 1, penalty = 3-1
#prediction is 1, where-as actually he belongs to risk bucket 2, penalty = (2-1)*2 
#since a risky person categorized as low-risk
#prediction is 1, where-as actually he belongs to risk bucket 3, penalty = (3-1)*2 
#since a risky person categorized as low-risk

#so penalty matrix is like this
tab
c(0,1,2,3,4, 2, 0, 1,2,3,4, 2, 0,1,2, 6,4,2,0,1, 8,6,4,2,0)->vec
length(vec)
penalty_matrix = matrix(data = vec, nrow = 5, ncol = 5, byrow = T)
penalty_matrix

#so if the predictions are as defined in the tab, then penalty will be calculated in the following manner:
penalty_matrix * tab -> penalty
sum(penalty)/nrow(Claims)

# <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- <- 
#what if we had chosen our regular baseline method of predicting the majority class
table(Claims$bucket2008)
#1 is the maority class
baseline_accuracy = 340202/nrow(Claims)
baseline_accuracy

#penalty of misclassification
table(predicted = rep(1, nrow(Claims)), actual = Claims$bucket2009)

#so predicted is a lower risk bucket, actually a higher risk bucket
penalty = 0*307444 + (2*87099) + (4*40976) + (6*19843) + (8*2643)
penalty/nrow(Claims)

#But we'll chose the above baseline accuracy and penalty because most likely to predict the 
#same risk buckets for the people. 
#------------------------------------------------------------------------------------

#train-test split
library(caTools)
set.seed(100)
spl = sample.split(Y = Claims$bucket2009, SplitRatio = .7)
training = Claims[spl, ]
test = Claims[!spl, ]

head(training)

#-------------------------------------------------------------------------------------------------------
library(rpart)
library(rpart.plot)
library(caret)

# Hyper-parametertuning of dec trees using 10-fold cv to obtain the best value of cp for pruning, trying to optimize on Kappa score
numfolds<-trainControl(method="cv",number=10) #conducting 10-cross validation
# Range of cp values, change it to find best cp value
cpGrid<-expand.grid(cp=seq(0.001,0.005,0.001))
# Building the model with cv using caret library
model_cv<-train(as.factor(bucket2009) ~ age + alzheimers + arthritis + cancer + 
                  copd + depression + diabetes + heart.failure + ihd + 
                  kidney + osteoporosis + stroke + bucket2008 + 
                  reimbursement2008, data=training, method="rpart", metric = "Kappa",
                trControl=numfolds,tuneGrid=cpGrid)
?train
model_cv
plot(model_cv)
model_cv$bestTune

#------------------------------------------------------------------------------------

#Hyper Parameter tuning for decision trees 
names(getModelInfo("rpart+", regex = T))
modelLookup("rpart")

model<-train(as.factor(bucket2009) ~., data=training, method="rpart", 
             parms=list(loss=penalty_matrix), trControl = numfolds, 
             tuneGrid = cpGrid, metric = "Accuracy")
model$bestTune

#-----------------------------------------------------------------------------------------
#Make the predictions
predict(model, newdata = test,type = "raw")->ypred
table(actual = test$bucket2009, predicted = ypred)->t
sum(t*penalty_matrix)
error = 1/nrow(test)
error
accuracy