library(tidyverse)
library(caret)

df.digits <- data.table::fread('data/train.csv')

# Function to Display a selected digit row
vect <- function(x) {as.numeric(as.vector(x))} #Turn Row of data frame into Vector
rotate <- function(x) {t(apply(x,2,rev))} # Rotate Matrix by 90 Degree

print_digit <- function(digits){
  opar <- par(no.readonly=TRUE) # Save Par
  
  # Set Par to produce grid of results
  x <- ceiling(sqrt(length(digits)))
  par(mfrow=c(x, x), mar=c(.1, .1, .1, .1))
  
  for (i in digits){
    d <- matrix(vect(df.numbers[i,-1]), byrow=TRUE, nrow=28)
    image(rotate(d), col=grey.colors(255), axes=FALSE)
    text(0.1,0.1,col='white',df.numbers[i,1])
  }
  par(opar) # Reset Par
}

df.digits <- df.digits[sample(nrow(df.digits))] # Shuffle Data

nzv <- nearZeroVar(df.digits) - 1; # Check for Near Zero Variance Columns

# Sample 80% of digits for training
samp <- sample(nrow(df.digits), nrow(df.digits)*0.8)
df.train <- df.digits[samp,]; df.test <- df.digits[-samp,]

X_train <- as.matrix(df.train[,-1]) # Matrix of Training Data
Y_train <- as.matrix(df.train[,1]) # Labels
y <- matrix(0, nrow(X_train), nrow(unique(Y_train)))
for (i in 1:nrow(X_train)){y[i, Y_train[i]+1] <- 1}; Y_train <- y
X_train <- X_train[,-nzv] # Dimension Reduction on Columns
X_train <- X_train/max(X_train) # Normalize

X_test <- as.matrix(df.test[,-1]) # Matrix of Testing Data
Y_test <- as.matrix(df.test[,1]) # Testing Labels
y <- matrix(0, nrow(X_test), nrow(unique(Y_train)))
for (i in 1:nrow(X_test)){y[i, Y_test[i]+1] <- 1}; Y_test <- y
X_test <- X_test[,-nzv] # Dimension Reduction on Columns
X_test <- X_test/max(X_test) # Normalize

# Transpose Matrices
X_train <- t(X_train)
Y_train <- t(Y_train)
X_test <- t(X_test)
Y_test <- t(Y_test)

# Activation Functions
ReLU_deriv <- function(x){ifelse(x<=0,0,1)}
sigmoid <- function(x){return(1 / (1 + exp(-x)))}

# Function to Initialize Weights
init <- function(X, Y, u){
  x <- nrow(X)
  k <- nrow(Y)
  
  # Initialize Hidden Layer Weights
  W1 <- 0.01 * matrix(rnorm(u * x), nrow = u, ncol = x, byrow = TRUE)
  b1 <- matrix(0, nrow=u,ncol=1)
  
  # Initialize Output Layer Weights
  W2 <- 0.01 * matrix(rnorm(k * u), nrow = k, ncol = u, byrow = TRUE)
  b2 <- matrix(0, nrow=k,ncol=1)
  
  return (list('W1'=W1, 
               'b1'=b1, 
               'W2'=W2, 
               'b2'=b2))
}

# Neural Network Function
NN <- function(X, Y, iterations, u=10, alpha=0.3){
  x <- nrow(X)
  k <- nrow(Y)
  m <- ncol(X)
  weight <- init(X, Y, u)
  W1=weight$W1; b1=weight$b1; W2=weight$W2; b2=weight$b2
  
  
  for (i in 1:iterations) {
    # Forward Propagation
    Z1 <- W1 %*% X + matrix(rep(b1,m), nrow=u)
    A1 <- pmax(Z1,0)
    Z2 <- W2 %*% A1 + matrix(rep(b2,m), nrow=k)
    A2 <- sigmoid(Z2)
    
    # Cost
    logp <- (log(A2) * Y) + (log(1-A2) * (1-Y))
    cost <- -sum(logp/m)
    
    # Backward Propagation
    dZ2 <- A2 - Y
    dW2 <- 1/m * (dZ2 %*% t(A1))
    db2 <- matrix(1/m * sum(dZ2), nrow = k)
    
    dZ1 <- (t(W2) %*% dZ2) * ReLU_deriv(A1)
    dW1 <- 1/m * (dZ1 %*% t(X))
    db1 <- matrix(1/m * sum(dZ1), nrow = u)
    
    # Update Parameters
    W2 <- W2 - alpha*dW2
    b2 <- b2 - alpha*db2
    W1 <- W1 - alpha*dW1
    b1 <- b1 - alpha*db1
    
    if (i %% 10 == 0) cat("Iteration", i, " | Cost: ", cost, "\n")
  }
  return (list('W1'=W1, 
               'b1'=b1, 
               'W2'=W2, 
               'b2'=b2))
}

# Train Network and Generate Weights
result <- NN(X_train, Y_train, 1000, u=20)

prediction <- function(X, Y, u=10){
  x <- nrow(X)
  k <- nrow(Y)
  m <- ncol(X)
  W1=result$W1; b1=result$b1; W2=result$W2; b2=result$b2
  
  Z1 <- W1 %*% X + matrix(rep(b1,m), nrow=u)
  A1 <- apply(Z1, c(1,2), ReLU)
  Z2 <- W2 %*% A1 + matrix(rep(b2,m), nrow=k)
  A2 <- softmax(Z2)
  
  return (A2)
}

# Generate Prediction
predicted <- prediction(X_test, Y_test, u=20)

# Convert Prediction to Binary based on which had highest score
test_prediction<- apply(predicted, 2,function(x) ifelse(x == max(x),1,0))

calc_acc <- function(prediction, actual) {
  tab <- table(actual, prediction)
  acc <- (tab[1]+tab[4])/(tab[1]+tab[2]+tab[3]+tab[4]) # Accuracy
  rec <- tab[4]/(tab[4]+tab[3]) # Recall
  prec <- tab[4]/(tab[4] + tab[2]) # Precision
  
  cat("Accuracy = ", acc*100, "%.\n")
  cat("Precision = ", prec*100, "%.\n")
  cat("Recall = ", rec*100, "%.")
}

calc_acc(test_prediction, Y_test)
