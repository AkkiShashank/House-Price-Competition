setwd("/Users/shashankgupta/Desktop/kaggle/House Price/")

library(data.table)
library(xgboost)
library(Metrics)
library(Matrix)
library(mice)
library(dplyr)

TRAIN="train.csv"
TEST="test.csv"
SUBMISSION= "sample_submission.csv"

train=fread(TRAIN,showProgress = T)
test=fread(TEST,showProgress = T)
y_train=train$SalePrice

train$Id=NULL
train$SalePrice=NULL
test$Id=NULL

train_test = rbind(train, test)
ntrain=nrow(train)

features=names(train)

for(f in features){
  if(class(train_test[[f]])=="character"){
    levels=sort(unique(train_test[[f]]))
    train_test[[f]]=as.integer(factor(train_test[[f]],levels = levels))
  }
}

features_to_drop<-c("Utilities","LotFrontage","Alley","MasVnrType","MasVnrArea","BsmtQual",
                    "BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
                    "Electrical","FireplaceQu","GarageType","GarageYrBlt",
                    "GarageFinish","GarageQual","GarageCond","PoolQC",
                    "Fence","MiscFeature")

#MSZoning,Utilities,Exterior1st,Exterior2nd,BsmtFinSF1,BsmtFinSF2,BsmtUnfSF,
#TotalBsmtSF,BsmtFullBath,BsmtHalfBath,KitchenQual,Functional,GarageCars,
#GarageArea,SaleType

train_x=train_test[1:ntrain,-features_to_drop,with= FALSE]
test_x=train_test[(ntrain+1):nrow(train_test),-features_to_drop,with= FALSE]

train_x[] <- lapply(train_x, as.numeric)
test_x[]<-lapply(test_x, as.numeric)

#missing values
set.seed(144)
to_impute<-as.data.frame(test_x)
impute=to_impute[c("MSZoning","Exterior1st","Exterior2nd","BsmtFinSF1",
                "BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath",
                "KitchenQual","Functional","GarageCars","GarageArea","SaleType")]
imputed=complete(mice(impute,m=5))

to_impute$MSZoning=imputed$MSZoning
to_impute$Utilities=imputed$Utilities
to_impute$Exterior1st=imputed$Exterior1st
to_impute$Exterior2nd=imputed$Exterior2nd
to_impute$BsmtFinSF1=imputed$BsmtFinSF1
to_impute$BsmtFinSF2=imputed$BsmtFinSF2
to_impute$BsmtUnfSF=imputed$BsmtUnfSF
to_impute$TotalBsmtSF=imputed$TotalBsmtSF
to_impute$BsmtHalfBath=imputed$BsmtHalfBath
to_impute$BsmtFullBath=imputed$BsmtFullBath
to_impute$KitchenQual=imputed$KitchenQual
to_impute$Functional=imputed$Functional
to_impute$GarageCars=imputed$GarageCars
to_impute$GarageArea=imputed$GarageArea
to_impute$SaleType=imputed$SaleType

test_x=as.data.table(to_impute)

dtrain=xgb.DMatrix(as.matrix(train_x),label= y_train)
dtest=xgb.DMatrix(as.matrix(test_x))

xgb_params = list(
  seed = 0,
  colsample_bytree = 0.5,
  subsample = 0.8,
  eta = 0.04, 
  objective = 'reg:linear',
  max_depth = 12,
  alpha = 1,
  gamma = 2,
  min_child_weight = 1,
  base_score = 7.76
)

xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

best_n_rounds=2000

gb_dt=xgb.train(xgb_params,dtrain,nrounds = as.integer(best_n_rounds))
submission=fread(SUBMISSION,colClasses = c("integer","numeric"))
submission$SalePrice=predict(gb_dt,dtest)
write.csv(submission,"xgb3.csv",row.names = FALSE)
