#####################################################################
## Deep Learning Malaysia Meetup                                   ##
## 15th February 2017                                              ##
## Sourcecode: https://github.com/kuanhoong/deeplearning-malaysia/ ##
## Title: Machine and Deep Learning with R                         ##
## Presenter: Poo Kuan Hoong                                       ##
#####################################################################

setwd('C:/Users/Kuan/Google Drive/DL-Meetup/')

library(data.table)

#load the training dataset with 42000 rows and 785 columns
train <- fread ( "data/train.csv")

# Create a 28*28 matrix with pixel color values
m = matrix(unlist(train[10,-1]), nrow = 28, byrow = TRUE)

# Plot that matrix
image(m,col=grey.colors(255))

# reverses (rotates the matrix)
rotate <- function(x) t(apply(x, 2, rev)) 

# Plot and show some of images
par(mfrow=c(2,3))
lapply(1:6, 
       function(x) image(
         rotate(matrix(unlist(train[x,-1]),nrow = 28, byrow = TRUE)),
         col=grey.colors(255),
         xlab=train[x,1]
       )
)

par(mfrow=c(1,1)) # set plot options back to default

#########################################################
## Training and Modelling                              ##
#########################################################

#load caret library
library (caret)

#split the dataset to 80% training and 20% testing
inTrain<- createDataPartition(train$label, p=0.8, list=FALSE)
training<-train[inTrain,]
testing<-train[-inTrain,]

#store the datasets into .csv files
write.csv (training , file = "train-data.csv", row.names = FALSE) 
write.csv (testing , file = "test-data.csv", row.names = FALSE)

#load h2o library
library(h2o)

#start a local h2o cluster
local.h2o <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, nthreads=-1)

#read the csv files
training <- read.csv ("train-data.csv") 
testing  <- read.csv ("test-data.csv")

# convert digit labels to factor for classification
training[,1]<-as.factor(training[,1])

# pass dataframe from inside of the R environment to the H2O instance
trData<-as.h2o(training)
trData[,1]<-as.factor(trData[,1])
tsData<-as.h2o(testing)
tsData[,1]<-as.factor(tsData[,1])

#measure start time
start<-proc.time()

#deep learning model
set.seed(1234)
model.dl <- h2o.deeplearning(x = 2:785,
                             y = 1,
                             trData,
                             activation = "RectifierWithDropout",
                             input_dropout_ratio = 0.2,
                             hidden_dropout_ratios = c(0.5,0.5),
                             balance_classes = TRUE, 
                             hidden = c(800,800),
                             epochs = 500)
#measure end time
end <- proc.time()

#time difference
diff=end-start
print(diff)

#use model to predict testing dataset
pred.dl<-h2o.predict(object=model.dl, newdata=tsData[,-1])
pred.dl.df<-as.data.frame(pred.dl)

summary(pred.dl,exact_quantiles=TRUE)
test_labels<-testing[,1]

#calculate number of correct prediction
sum(diag(table(test_labels,pred.dl.df[,1])))

# shut down virtual H2O cluster
h2o.shutdown(prompt = FALSE)