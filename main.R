# Case-Challenge for https://github.com/Fiddleman/BusinessAnalytics
# author: Lukas Leist

# Install and Import Packages
## Install
install.packages("party")
install.packages("devtools")
devtools::install_github("rstudio/keras")
library(keras)
install_keras()


## Import
library("party")
library(keras)


# Load Data
## Training Data
train = read.csv("data/train.csv", row.names="PassengerId")

## Testing Data
test = read.csv("./data/test.csv")
table(is.na(train$Embarked))
colnames(train)


# Clean and Explore
## Function for converting columns to factors and feature engineering
## Missing Values for Age, Deck and Embarked get Estimated
prepare = function(dataset) {
  if("Survived" %in% colnames(dataset)) {
    dataset$Survived = factor(dataset$Survived, labels = c("died", "survived"))  
  }
  dataset$Pclass = factor(dataset$Pclass, labels = c("1st", "2nd", "3rd"))
  dataset$Sex = factor(dataset$Sex, levels = c("female", "male"))
  dataset$Embarked = factor(dataset$Embarked, levels = c("C", "Q", "S"), labels = c("Cherbourg", "Queenstown", "Southhampton"))
  
  #Extract Deck from starting letter of cabin and estimate missing
  
  dataset$Deck = factor(substr(dataset$Cabin, 1,1), levels = c("A", "B", "C", "D", "E", "F", "G", "T"))
  #print(summary(dataset$Deck))
  if("Deck" %in% colnames(train)) {
    deck.tree = ctree(Deck ~ Fare + Pclass + Embarked, data = train[!is.na(dataset$Deck),])
  }
  else {
    deck.tree = ctree(Deck ~ Fare + Pclass + Embarked, data = dataset[!is.na(dataset$Deck),])
  }
  dataset[is.na(dataset$Deck),]$Deck = predict(deck.tree, dataset[is.na(dataset$Deck),])
  #print(summary(dataset$Deck))
  
  dataset$Family.Members = dataset$Parch + dataset$SibSp
  
  estimator.age = lm(Age ~ Parch + SibSp, data=train)
  #print("Age estimation:")
  #print(summary(dataset$Age))
  estimations = predict(estimator.age, dataset[is.na(dataset$Age),])
  estimations[estimations < 0] = 0
  dataset[is.na(dataset$Age),]$Age = estimations
  #print(summary(dataset$Age))
  
  if(TRUE %in% is.na(dataset$Embarked)) {
    dataset[is.na(dataset$Embarked), "Embarked"] = "Southhampton"
  }
  
  if(TRUE %in% is.na(dataset$Fare)) {
    dataset[is.na(dataset$Fare), "Fare"] = median(train[!is.na(train$Fare), "Fare"])
  }
  
  return(dataset)
}


colnames(train)
train = prepare(train)
summary(train)

test = prepare(test)
summary(test)

## Explore Datasets

sex.table = table(train$Survived, train$Sex)
summary(sex.table)

Pclass.table = table(train$Survived, train$Pclass)
summary(Pclass.table)

Embarked.table = table(train$Survived, train$Embarked)
summary(Embarked.table)

Deck.table = table(train$Survived, train$Deck)
summary(Deck.table)

# Modelling: Decision Tree Based Approach

## Fitting Decision Tree
predictor = ctree(Survived ~ Pclass + Sex + Age + Family.Members + Fare + Embarked + Deck, data=train)

predictor

## Predicting
predictions = predict(predictor, test)
summary(predictions)

# Moddeling: Neuronal Network Approach


prepare.y = function(col) {
  tmp = as.numeric(col) - 1
  tmp = to_categorical(tmp, length(levels(col)))
  return(tmp)
}


prepare.x = function(dataset) {
  cbind(
    prepare.y(dataset$Pclass),
    prepare.y(dataset$Sex),
    dataset$Age,
    dataset$SibSp,
    dataset$Parch,
    dataset$Fare,
    prepare.y(dataset$Deck),
    prepare.y(dataset$Embarked)
  )
}

summary(train$Deck)

train.y = prepare.y(train$Survived)
train.x = prepare.x(train)
test.x = prepare.x(test)


model <- keras_model_sequential()
model %>% 
  layer_dense(units = 32, activation = 'relu', input_shape = c(20)) %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  train.x, train.y, 
  epochs = 4000
)

predictions = model %>% predict_classes(test.x)
predictions = factor(predictions + 1, labels = c("died", "survived"))
summary(as.factor(predictions))

# Export
result = data.frame(PassengerId = test$PassengerId)
result$Survived = as.numeric(predictions) - 1

table(result$Survived)
write.csv(result, file = "./results/result.csv", quote = FALSE, row.names = FALSE)
