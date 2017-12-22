###############################################################################################
# Analyze embeddings extracted from the vgg16 architecture
# trained on imageNet applied to the mit67 data set.   
# Lecturer: Prof. Dario Garcia
# November 22, 2017  
#
# Data set: 
# 
#
#
# Jakob Gerstenlauer
# jakob.gerstenlauer@gjakob.de
###############################################################################################

#remove old objects for safety resons
rm(list=ls(all=TRUE))

#proportion of observations included in training set
train.prop<-0.3333

#utility function
glue<-function(...){paste(...,sep="")}

#define path of standard directories
workingDir<-"~/Documents/UPC/2017/02/DL/labs/03/code/_codes/3.Embeddings/Image_Emb"
plotDir<-glue(workingDir,"/plots")
dataDir <- workingDir

#set working directory
setwd(dataDir)

#set seed to make analysis reproducible (deterministic)
set.seed(123)

#*****************************************************************************
#Step 1: Read the embeddings. Select 50% of the observations as training data.
#*****************************************************************************

#read the complete training data set from a standard tab delimited ascii text file
d<-read.table("Embeddingmit67_embeddings_fc1.npz.csv", sep=";", header=FALSE)
dim(d)
#[1] 5360 4096

#read the labels
l<-read.table("Labelsmit67_embeddings_fc1.npz.csv", sep=";", header=FALSE)
dim(l)
#[1] 5360    1
#The number of rows and columns is correct! 

#Add the labels to the data set:
d<-cbind(d,l)
dim(d)
#[1] 5360 4097

#create informative column names:
names(d)<-c(glue("dim",1:4096), "class")

#Now I restrict the observations to 50% of the original data set:
#I have to use a stratified sample in regard to the labels (classes).
#https://stackoverflow.com/questions/23479512/stratified-random-sampling-from-data-frame-in-r

table(d$class)
d$class <- as.factor(d$class)
classLabels <- levels(d$class)

library(splitstackshape)
set.seed(123456)
listOfTwoDataTables <- stratified(d, "class", size=train.prop, replace = FALSE, bothSets = TRUE)
str(listOfTwoDataTables)
d.train <- listOfTwoDataTables$SAMP1
d.test <- listOfTwoDataTables$SAMP2

dim(d.train)
#[1] 1787 4097
dim(d.test)
#[1] 3573 4097

#*****************************************************************************
#Step 2: Define the response matrix (Y) and the predictor matrix (X). 
# Center the predictor matrix.
#*****************************************************************************

X <- d.train[,1:4096]

#We have to convert Y into a matrix with 10 vectors, 
#each vector being a dummy variable for one of the 67 possible classes.
#https://stackoverflow.com/questions/13901153/converting-r-factors-into-binary-matrix-values#13901840
Y <- model.matrix( ~ 0 + class, d.train)
dim(Y)
#[1] 1787   67

#assign useful names to the column names:
dimnames(Y)[[2]]<-classLabels
Y.mean<-apply(Y,2,mean)

#Center the matrix of predictors
#scaling does not make sense 
X.mean<-apply(X,2,mean)
Xs <- as.matrix(scale(X, center = TRUE, scale = FALSE))


#*****************************************************************************
# Step 3:	Perform a PLSR2 using no validation.
# Decide how many components you retain for prediction based on mean RSquared.
#*****************************************************************************

library(pls)
m1.pls2 <- plsr(Y ~ Xs, validation = "none")
#summary(m1.pls2)
setwd(dataDir)
save(m1.pls2, file="m1.pls2.Standardized")

# Plot of R2 for each digit 
#setwd(plotDir)
#jpeg("R2_all_labels.jpeg")
#plot(R2(m1.pls2), legendpos = "bottomright")
#dev.off()

#Inspect the object:
dim(R2(m1.pls2)$val[1,,])
#[1]  67 1787
#rows: response
#columns: cumulative number of components

#calculate the mean R2 over all digits for an increasing number of components:
r2.mean<-apply(R2(m1.pls2)$val[1,,],2,mean)

barplot(r2.mean[1:500])
#It seems that 500 components are sufficient.

#zoom in:
setwd(plotDir)
jpeg("NrOfPLSIIComponents.jpeg")
plot(1:1000,r2.mean[1:1000],type="l",xlab="number of components", ylab="coefficient of determination")
abline(v=500)
text("500 components",x=620,y=0.1)
dev.off()
#Conclusion: 500 components are sufficient! 


###################################################################################################################
# 4.	Predict the responses in the test data, be aware of the appropriate centering. 
# Compute the average R2 in the test data.
###################################################################################################################

dim(d.test)
#[1] 3573 4097
#We have 3573 observations and 4096 dimensions of the embedding and finally the class

#create informative column names:
names(d.test)<-c(glue("dim",1:4096), "class")

#Define the response matrix (Y) and the predictor matrix (X).
X.test <- as.matrix(d.test[,1:4096])
#subtract the means of the training set:
#numRows<-dim(X.test)[1]
#numCols<-dim(X.test)[2]
#X_means<- matrix(rep(X.mean,numRows),byrow=TRUE,ncol=numCols)
#X.test <- X.test - X_means

#We have to convert Y into a matrix with 67 vectors, 
#each vector being a dummy variable for one of the 67 possible classes.
#https://stackoverflow.com/questions/13901153/converting-r-factors-into-binary-matrix-values#13901840
Y.test<-model.matrix( ~ 0 + class, d.test)
dim(Y.test)
#[1] 3573 67

#assign useful names to the column names:
dimnames(Y.test)[[2]]<-classLabels
Y.test.mean<-apply(Y.test,2,mean)

#Using the predict function:
predictions.test<-predict(m1.pls2, ncomp = 500, newdata = as.matrix(X.test))
dim(predictions.test)
dim(predictions.test)<-c(3573,67)

#If I only consider the 50 first axes:
#predictions.test.50<-predict(m1.pls2, ncomp = 50, newdata = as.matrix(X.test))
#dim(predictions.test.50)
#dim(predictions.test.50)<-c(3573,67)

############################################################################################
#Calculate the coefficient of determination
#
#Here we use three different methods to calculate the R-Square.
#A) as % of the variance unexplained in a global mean model.
#B) as % of the variance unexplained in a class mean model.
#C) as the arithmetic mean of the % of the variance unexplained in a class mean model.
############################################################################################

calculateSquaredErrors<-function(x,y){
  x<-as.vector(x)
  y<-as.vector(y)
  sum((x-y)**2)
}

Y.pred<-predictions.test
#Y.pred.50 <- predictions.test.50

#*****************************************************************************
#Step 5:
# Assign every test individual to the maximum response
# and compute the error rate.
#*****************************************************************************

findIndexMax<-function(x){
  which(x==max(x))
}

#a simple vector with the predicted class for each observation
#-1 because first class is digit zero
predicted_class<-apply(Y.pred, 1, findIndexMax)
#predicted_class_50<-apply(Y.pred.50, 1, findIndexMax)

#the real class
empirical_class <- as.numeric(d.test$class)

#Let`s plot the confusion matrix:
confusionMatrix <- table(predicted_class, empirical_class)
correctPredictions<-sum(diag(confusionMatrix))
Accuracy<-correctPredictions/sum(confusionMatrix)
#[1] 0.5242093 (results without standardization
#[1] 0.5320459 (results with standardization of both input and output matrix)

#confusionMatrix.50 <- table(predicted_class_50, empirical_class)
#correctPredictions.50<-sum(diag(confusionMatrix.50))
#Accuracy<-correctPredictions.50/sum(confusionMatrix.50)
#0.5166527

#############################################
#Calculate the loadings for the training data:

#Scores of the individuals for the training data
scores<-m1.pls2$scores[,1:500]

#Loadings of the original features for the training data
loadings<-m1.pls2$loadings[,1:500]

#create a new data set containing the scores and the class
scores.train<-as.data.frame(scores)
scores.train$class<-d.train$class

############# regression tree ############
library(rpart)
set.seed(567)
#Use Gini index as impurity criterion:
m1.rp <- rpart(class ~ ., method="class", data=scores.train, 
               control=rpart.control(cp=0.001, xval=10))
printcp(m1.rp)

setwd(plotDir)
jpeg("CrossValidatedPredictionErrorRegressionTree.jpeg")
plotcp(m1.rp)
dev.off()

m2.rp<-prune(m1.rp, cp = 0.0015)
plotcp(m2.rp)
printcp(m2.rp)
#105 nodes seems to be appropriate! 

############# random forest  ############
set.seed(9019)
#install.packages("randomForest")
library(randomForest)

#Possible hyperparameter:
# ntree: Number of trees to grow. This should not be set to too small a number, to ensure that every input row gets predicted at least a few times.
# mtry:	 Number of variables randomly sampled as candidates at each split. Note that the default values are different for classification (sqrt(p) where p is number of variables in x) and regression (p/3)
# classwt:  Priors of the classes. Need not add up to one. Ignored for regression.
# strata: Maybe define the subject (the person) as strata 
#A (factor) variable that is used for stratified sampling.
# sampsize	
# Size(s) of sample to draw. For classification, if sampsize is a vector of the length
# the number of strata, then sampling is stratified by strata, 
# and the elements of sampsize indicate the numbers to be drawn from the strata.
# nodesize: use 10!	
# Minimum size of terminal nodes. Setting this number larger causes smaller trees to be grown (and thus take less time). Note that the default values are different for classification (1) and regression (5).
# maxnodes	

X <- scores 
#Now I have to calculate the PLSII axes for the test data:
Xt <- as.matrix(X.test)  %*% loadings 
Xt50 <- as.matrix(X.test)  %*% loadings[,1:50]
Yt <- as.factor(d.test$class)

#What happens if I only consider the first 50 axes and restrict to 67 nodes?
m1.rf <- randomForest(y=as.factor(d.train$class),
                      x=X, 
                      ntree=1000, 
                      #classwt= rep(1/6,6), 
                      importance=TRUE, 
                      xtest=Xt, 
                      ytest=Yt, 
                      nodesize=5, 
                      maxnodes=67)

#confusion table for the training set:
Ctrain<-m1.rf$confusion
sum(diag(Ctrain))/sum(Ctrain)
#[1] 0.7828302

#confusion table for the test set:
Ctest<-m1.rf$test$confusion
sum(diag(Ctest))/sum(Ctest)
#[1] 0.3242119
# 
# #Plot the variable importance of the ten most important variables
# var.imp<-as.data.frame(m1.rf$importance)
# sort(var.imp$MeanDecreaseAccuracy, decreasing=TRUE)[1:10]
# row.names(var.imp[with(var.imp, order(-MeanDecreaseAccuracy)), ])[1:5]
# 
# #Plot the error rate for an increasing number of trees:
# setwd(plotDir)
# jpeg("ErrorRate_NrOfTrees.jpeg")
# plot(m1.rf)
# dev.off()
# 
# print(m1.rf)
# m1.rf.summary<-summary(m1.rf)
# str(m1.rf.summary)
setwd(dataDir)
save.image(file = "DL_Lab3.RData")