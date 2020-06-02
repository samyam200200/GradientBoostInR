#Loading dependencies
library(tidyverse)
library(magrittr)
library(vtreat)
library(xgboost)
library(gmodels)
#WisconsinBreastCancerData
url<-"https://assets.datacamp.com/production/course_6430/datasets/WisconsinCancer.csv"
wbcd_xg<-read.csv(url, header = T, sep=",")
#removing patient id and X column
wbcd_xg<-wbcd_xg[c(-1,-33)]
#extracting variable names for treatment
vars<-names(wbcd_xg[2:31])
#randomly sampling 80% data set
set.seed(100)
wbcd_xgtrain<-sample_n(wbcd_xg, size=80/100*nrow(wbcd_xg))
wbcd_xgtest<-anti_join(wbcd_xg,wbcd_xgtrain)
#Changing the benign and malignant to 0,1 respectively for better fit onto the model
wbcd_xgtrain$diagnosis<-as.numeric(factor(wbcd_xgtrain$diagnosis))
wbcd_xgtrain$diagnosis<-ifelse(wbcd_xgtrain$diagnosis==1, 0,1)
wbcd_xgtest$diagnosis<-as.numeric(factor(wbcd_xgtest$diagnosis))
wbcd_xgtest$diagnosis<-ifelse(wbcd_xgtest$diagnosis==1, 0,1)
# preparing training variables so that data has fewer exceptional cases, making it easier to safely use models in production. 
treatment__training<-designTreatmentsZ(wbcd_xgtrain, vars)
#scoreFrame describes variable mapping and types
scoreFrame<-treatment__training%>%use_series(scoreFrame)%>%select(varName, origName, code)
#getting new names of clean and lev variables
newvars<-scoreFrame%>%filter(code %in% c("clean", "lev"))%>%use_series(varName)
#final prepared training data for modelling
final_treatment_training<-prepare(treatment__training, wbcd_xgtrain, varRestriction = newvars)
# preparing testing variables so that data has fewer exceptional cases, making it easier to safely use models in production. Similar process applied to training data as well
treatment__testing<-designTreatmentsZ(wbcd_xgtest, vars)
scoreFrame<-treatment__testing%>%use_series(scoreFrame)%>%select(varName, origName, code)
newvars<-scoreFrame%>%filter(code %in% c("clean", "lev"))%>%use_series(varName)
final_treatment_testing<-prepare(treatment__testing, wbcd_xgtest, varRestriction = newvars)
#Running xgb.cv with large number of trees. Recording error mean and using the respective number of trees that minimizes error
cv<-xgb.cv(data=as.matrix(final_treatment_training), #inputting training data as matrix
           label = wbcd_xgtrain$diagnosis, #outcome variables
           nrounds = 100, #number of trees to fit
           nfold=5, #number of folds for cross validation
           objective = "binary:logistic", #logistic decision
           eta=0.3, #learning rate
           max_depth=6, #maximum number of depths for individual tree
           early_stopping_rounds = 10, 
           verbose=0)
elog<-cv$evaluation_log
elog
#calculating error mean
elog%>%summarise(ntrees.train=which.min(elog$train_error_mean), ntrees.test=which.min(elog$test_error_mean))

#using test error mean as number of round
model_xgb <- xgboost(data = as.matrix(final_treatment_training),  label =wbcd_xgtrain$diagnosis,      nrounds =which.min(elog$test_error_mean),     objective = "binary:logistic", eta = 0.3,    depth = 6,   verbose = 0  )

#predicting the model
wbcd_xgtest$predicted<-round(predict(model_xgb, as.matrix(final_treatment_testing), type=response))
#calculating sensitivity and specificity using contingency table
CrossTable(x = wbcd_xgtest$diagnosis, y = wbcd_xgtest$predicted,
           prop.chisq=FALSE)
