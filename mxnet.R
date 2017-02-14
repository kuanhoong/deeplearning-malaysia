#####################################################################
## Deep Learning Malaysia Meetup                                   ##
## 15th February 2017                                              ##
## Sourcecode: https://github.com/kuanhoong/deeplearning-malaysia/ ##
## Title: Machine and Deep Learning with R                         ##
## Presenter: Poo Kuan Hoong                                       ##
#####################################################################

library(mxnet)
library(data.table)

setwd('C:/Users/Kuan/Google Drive/DL-Meetup/')

train <- fread('data/train.csv')
test <- fread('data/test.csv')

###########################
## Data Exploration      ##
###########################

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

##################################
## Training and Modelling       ##
##################################

train <- data.matrix(train)
test <- data.matrix(test)

train.x <- train[,-1]
train.y <- train[,1]

# Linearly transform the greyscale of each image of [0, 255] to [0,1]
# Transpose the input matrix to npixel x nexamples, which is the major # format for columns accepted by MXNet (and the convention of R).
train.x <- t(train.x/255)
test <- t(test/255)

# The number of each digit is fairly evenly distributed:
table(train.y)

# Configuring the Network
data <- mx.symbol.Variable("data")
# set the first hidden layer as fully connected with 128 hidden
# neurons.
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
# relu activation function
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
# Second hidden layer with 64 hidden neurons 
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
# relu activation function
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
# Third hidden layer with 10 hidden neurons 
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
#  softmax to get a probabilistic prediction
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

# Assign CPU to mxnet
devices <- mx.cpu()

mx.set.seed(1234)
model <- mx.model.FeedForward.create(softmax,
                                     X=train.x,
                                     y=train.y,
                                     ctx=devices,
                                     num.round=10,
                                     array.batch.size=100,
                                     learning.rate=0.07,
                                     momentum=0.9,
                                     eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))

# Predict test dataset
preds <- predict(model, test)
dim(preds)

# To extract the maximum label for each row, use max.col:
pred.label <- max.col(t(preds)) - 1
table(pred.label)

# modify the .csv format for submission to Kaggle
submission <- data.frame(ImageId=1:ncol(test), Label=pred.label)
write.csv(submission, file='submission.csv', row.names=FALSE,  quote=FALSE)


######################################
## Le-Net                           ##
######################################

# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

#reshape data
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test
dim(test.array) <- c(28, 28, 1, ncol(test))

#compare performance
n.gpu <- 1
device.cpu <- mx.cpu()
device.gpu <- lapply(0:(n.gpu-1), function(i) {
  mx.gpu(i)
})

#train model on CPU
mx.set.seed(0)
tic <- proc.time()
model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=device.cpu, num.round=1, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))

print(proc.time() - tic)

#train model on GPU
mx.set.seed(0)
tic <- proc.time()
model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=device.gpu, num.round=5, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))

print(proc.time() - tic)

# Predict test dataset
preds <- predict(model, test.array)
# To extract the maximum label for each row, use max.col:
pred.label <- max.col(t(preds)) - 1

# modify the .csv format for submission to Kaggle
submission <- data.frame(ImageId=1:ncol(test), Label=pred.label)
write.csv(submission, file='submission1.csv', row.names=FALSE, quote=FALSE)
