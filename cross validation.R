# ====================================================================================================
# author:   ahmed nagy radwan
# ====================================================================================================
library(SnowballC)
library(rpart)
library(caret)
library(tm)
library(MASS)
#======================================================================================================
dir  = getwd()

setwd(dir = dir)


load("dataset.RData")
data.matrix <- all_data
remove(all_data)
data.matrix$content <-iconv(data.matrix$content,"WINDOWS-1252","UTF-8")
#-----------------------------------------------------------------------------------------------------

preProcess_TFIDF <- function(row.data, stopword.dir, BagOfWord , boolstemm ){
  
  packages <- c('tm')
  if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
    install.packages(setdiff(packages, rownames(installed.packages())))  
  }
  
  library(tm)
  row.data<-iconv(row.data,"WINDOWS-1252","UTF-8")
  
  stopwordlist<- readLines(stopword.dir)
  train_corpus<- Corpus(VectorSource(row.data))
  train_corpus <-tm_map(train_corpus,content_transformer(tolower))
  train_corpus<-tm_map(train_corpus,removeNumbers)
  train_corpus<-tm_map(train_corpus,removeWords,stopwords("english"))
  train_corpus<-tm_map(train_corpus,removeWords,stopwordlist)
  train_corpus<- tm_map(train_corpus,removePunctuation)
  train_corpus<-tm_map(train_corpus,stripWhitespace)
  
  if(boolstemm){
    
    train_corpus<-tm_map(train_corpus,stemDocument,language = "english")
  }
  
  DTM <-DocumentTermMatrix(train_corpus,
                           control = list(tolower = T ,
                                          removeNumbers =T ,
                                          removePunctuation = T ,
                                          stopwords = T 
                                          , stripWhitespace = T ,
                                          dictionary = BagOfWord,
                                          weighting = weightTfIdf))
  
  
  print("DTM DONE !!")
  print(DTM)
  return(DTM)
}
#----------------------------------------------------------------------------------------------------
train_dtm <-preProcess_TFIDF(row.data = data.matrix$content ,stopword.dir = "stopword.txt",
                             BagOfWord = NULL , TRUE)
train_dtm <- removeSparseTerms(train_dtm,0.993)
BagOW <- findFreqTerms(train_dtm)

dim(train_dtm)

train_matrix <- as.matrix(train_dtm)
#train_matrix <- binary.weight(train_matrix)
train_data_model <- as.data.frame(train_matrix)

train_data_model <- data.frame(y=data.matrix$lable , x = train_data_model)
#----------------------------------------------------------------------------------------------------
remove(bbc.data.matrix)
remove(train_dtm)
remove(train_matrix)
#----------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------
library(caret)
library(randomForest)
library(rpart)
library(e1071)
library(doParallel)
acc_matrix <- matrix(nrow = 10 , ncol = 8 )
colnames(acc_matrix) <- c("naive-bayes-accuracy","decision-tree-accuracy",
                          "random-forest-accuracy " , "support-vector-machine-accuracy",
                          "naive-bayes-time","decision-tree-time","random-forest-time" , "support-vector-machine-time" )


#Randomly shuffle the data
train_data_model<-train_data_model[sample(nrow(train_data_model)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(train_data_model)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==8,arr.ind=TRUE)
  testData <- train_data_model[testIndexes, ]
  trainData <- train_data_model[-testIndexes, ]
#---------------------------------------------------------------  
  cl <- makePSOCKcluster(5)
  registerDoParallel(cl)
  timr <- Sys.time()
  
  svm <- train(y~.,data = trainData,method = 'svmLinear3')
  
  ## When you are done:
  stopCluster(cl)
  
  total_runtime <- difftime(Sys.time(), timr)
  print(total_runtime)
  
  
  pre <- predict(svm,testData[,-1])
  
  t1 <- table(pre,testData$y)
  acc = sum(diag(t1))/sum(t1)
  print(acc)
  acc_matrix[i,4] = acc
  acc_matrix[i,8] = total_runtime
#------------------------------------------------------------------------------------------------
  timr <- Sys.time()
 naiveB_model = naiveBayes( y~.,data =trainData)
 total_runtime <- difftime(Sys.time(), timr)
 print(total_runtime)
 naiveB_testpred = predict(naiveB_model,testData[,-1])  
 t2 <- table(naiveB_testpred,testData$y)
 acc = sum(diag(t2))/sum(t2)
 print(acc)
 acc_matrix[i,1] = acc
 acc_matrix[i,5] = total_runtime
#------------------------------------------------------------------------------------------------ 
 timr <- Sys.time()
dtree_model <-rpart(y~.,trainData , method =  "class") 
 total_runtime <- difftime(Sys.time(), timr)
 print(total_runtime)
dtree_testpred = predict(dtree_model,newdata = testData[,-1] ,type = "class" )
 t3 <- table(dtree_testpred,testData$y)
 acc = sum(diag(t3))/sum(t3)
 print(acc)
 acc_matrix[i,2] = acc
 acc_matrix[i,6] = total_runtime
#------------------------------------------------------------------------------------------------ 
 timr <- Sys.time()
 rf_model <- randomForest(x = trainData[,-1],y = trainData$y , ntree = 60 )
 total_runtime <- difftime(Sys.time(), timr)
 print(total_runtime)
rf_testpred = predict(rf_model,newdata = testData[,-1] )
 t4 <- table(rf_testpred,testData$y)
 acc = sum(diag(t4))/sum(t4)
 print(acc)
 acc_matrix[i,3] = acc
 acc_matrix[i,7] = total_runtime
 
 }

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

acc_plot <- as.data.frame(acc_matrix[,1:4])

require(reshape2)
ggplot(melt(acc_plot), aes(x = variable, y = value, fill = variable)) +
  geom_boxplot() + 
  ggtitle("10 fold cross validation") + 
  theme(axis.text.x = element_text(angle=0, face="bold", colour="black"))
#--------------------------------------------------------------------------------------------------

