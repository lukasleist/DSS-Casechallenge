# Case-Challenge for https://github.com/Fiddleman/BusinessAnalytics
# author: Lukas Leist

# Load Data
## Training Data
train = read.csv("data/train.csv", row.names="PassengerId")

## Testing Data
test = read.csv("./data/test.csv")


# Clean and Explore
## Function for converting columns to factors and feature engineering
prepare = function(dataset) {
  if("Survived" %in% colnames(dataset)) {
    dataset$Survived = factor(dataset$Survived, labels = c("died", "survived"))  
  }
  dataset$Pclass = factor(dataset$Pclass, labels = c("1st", "2nd", "3rd"))
  dataset$Sex = factor(dataset$Sex, levels = c("female", "male"))
  dataset$Embarked = factor(dataset$Embarked, levels = c("C", "Q", "S"), labels = c("Cherbourg", "Queenstown", "Southhampton"))
  
  #Extract Deck from starting letter of cabin
  dataset$Deck = factor(substr(dataset$Cabin, 1,1), levels = c("A", "B", "C", "D", "E", "F", "G", "T"))
  dataset$Family.Members = dataset$Parch + dataset$SibSp
  return(dataset)
}


## Explore Datasets

colnames(train)
train = prepare(train)
summary(train)

test = prepare(test)
summary(test)

sex.table = table(train$Survived, train$Sex)
summary(sex.table)

Pclass.table = table(train$Survived, train$Pclass)
summary(Pclass.table)

Embarked.table = table(train$Survived, train$Embarked)
summary(Embarked.table)

Deck.table = table(train$Survived, train$Deck)
summary(Deck.table)

# Modelling: Decision Tree Based Approach
install.packages("party")
library("party")

## Fitting Decision Tree
predictor = ctree(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Deck, data=train)

## Predicting
predictions = predict(predictor, test)
summary(predictions)

# Export: Decision Tree Based Approach
result = data.frame(PassengerId = test$PassengerId)
result$Survived = as.numeric(predictions) - 1

table(result$Survived)
write.csv(result, file = "./result.csv", quote = FALSE, row.names = FALSE)

# Moddeling: Neuronal Network Approach
##
install.packages("devtools")
devtools::install_github("rstudio/keras")
library(keras)
install_keras()


prepare.y = function(col) {
  tmp = as.numeric(col) - 1
  tmp = to_categorical(tmp, length(levels(col)))
  return(tmp)
}

prepare.x = function(dataset) {
  cbind(as.numeric(dataset$Pclass), as.numeric(dataset$Sex), dataset$Age, dataset$SibSp, dataset$Parch, dataset$Fare, as.numeric(dataset$Deck))
}

train.y = prepare.y(train$Survived)
train.x = prepare.x(train)
train.x[is.na(train.x)] = 0
test.x = prepare.x(test)

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 32, activation = 'relu', input_shape = c(7)) %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  train.x, train.y, 
  epochs = 30
)

model %>% predict_classes(test.x)
