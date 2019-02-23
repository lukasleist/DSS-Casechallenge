# Case-Challenge for https://github.com/Fiddleman/BusinessAnalytics
# author: Lukas Leist

# load Dataset
train = read.csv("data/train.csv",
                 row.names="PassengerId")

# Clean and describe dataset
## Convert columns to factors
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

colnames(train)
train = prepare(train)
summary(train)


# Exploration of Data
sex.table = table(train$Survived, train$Sex)
summary(sex.table)

Pclass.table = table(train$Survived, train$Pclass)
summary(Pclass.table)

Embarked.table = table(train$Survived, train$Embarked)
summary(Embarked.table)

Deck.table = table(train$Survived, train$Deck)
summary(Deck.table)

# Decision tree based Approach
## Fitting Decision Tree
install.packages("party")
library("party")
colnames(train)
predictor = ctree(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Deck, data=train)

# Neuronal Network Approach
##
install.packages("keras")
library(keras)
install_keras()

train.y = train$Survived
train.x = train[c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Deck")]


# Load Test Data
test = read.csv("./data/test.csv")
test = prepare(test)

summary(test)

# Predict Results
predictions = predict(predictor, test)
summary(results)

result = data.frame(PassengerId = test$PassengerId)
result$Survived = as.numeric(predictions) - 1

table(result$Survived)

write.csv(result, file = "./result.csv", quote = FALSE, row.names = FALSE)
