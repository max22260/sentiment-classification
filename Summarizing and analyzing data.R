#-----------------------------------------------------------------------------------------------------
# author:   ahmed nagy radwan
#-----------------------------------------------------------------------------------------------------


# dataset : https://www.kaggle.com/c/sentiment-classification-on-large-movie-review/data

#----------------------- install packeges -------------------
install.packages("lubridate")
install.packages("textreadr")
install.packages("wordcloud")
install.packages("rpart")
install.packages("caret")
install.packages("tm")
install.packages("MASS")
install.packages("RWeka")
install.packages("rminer")
install.packages("kernlab")
install.packages("SnowballC")
# 
############################# load packeges
library(SnowballC)
library(textreadr)
library(wordcloud)
library(rpart)
library(caret)
library(tm)
library(MASS)
library(RWeka)
library(rminer)
library(kernlab)

#--------------------------------------read Data---------------------------------------------#
## the read fun to read all files *.txt from  directory                                      #
pos_data <-read_dir('C:\\Users\\max22\\Desktop\\txt_mining\\train\\pos') #read positive data # 
Neg_data <-read_dir('C:\\Users\\max22\\Desktop\\txt_mining\\train\\neg') #read negative data #
                                                                                             #
#--------------------------------------------------------------------------------------------#

setwd("Desktop/spring2018/txt_mining/")

#--------------------------------------read Data---------------------------------------------#
## the read fun to read all files *.txt from  directory                                      #
pos_data <-read.dir("train/pos",".txt") #read positive data # 
Neg_data <-read.dir('train/neg/',".txt") #read negative data #
#
#--------------------------------------------------------------------------------------------#


#-----------------------------write data As csv file ----------------------------------------#
pos = as.matrix(pos_data)                                                                    #
write.csv( x=pos ,file = "train/pos.csv")            #
                                                                                             #
neg = as.matrix(Neg_data)                                                                    #
write.csv( x=neg ,file = "train/neg.csv")            #
                                                                                             #
#--------------------------------------------------------------------------------------------#

#------------------------add lable and merge two csv file -----------------------------------#


pos_data_csv = read.csv("train/pos.csv", header=T) #read from csv file
neg_data_csv = read.csv("train/neg.csv", header=T)# read from csv file


pos_data_csv$X <- NULL #remove count 
neg_data_csv$X <- NULL #remove count 

################add lable ############
pos_data_csv$lable <- c("positive")
neg_data_csv$lable <- c("negative")

######################################## merge and save result #########################################
all_data = rbind(pos_data_csv[1:1500,],neg_data_csv[1:1500,])
str(all_data)
summary(all_data)
nrow(all_data)
prop.table(table(all_data$lable))
write.csv( x=all_data ,file = "train/all_data1.csv")
#---------------------------------------------------------------------------------------------

all_data <- read.csv(file = "train/all_data1.csv" ,header = T )
all_data$X <- NULL

all_data$lable  <- factor(all_data$lable)

#-------------------------------------------------------------------------------------------------------
#-------------------------------- summrize our data ----------------------------------------------------

str(all_data) #structure of data
summary(all_data) #sumrry for data
nrow(all_data) 
prop.table(table(all_data$lable)) # lable probability 

#-------------------------------------------------------------------------------------------------------

#--------------------------- prepare training_dataset and test1_dataset , test2_dataset -----------------
set.seed(100) # for randmnes

trian <- createDataPartition(y=all_data$lable,p=0.70 , list = FALSE)
train_dataset <- all_data[trian,]

testdata <- all_data[-trian,]
test_dataset <-createDataPartition(y=testdata$lable,p=0.50, list = FALSE)
test1 <- testdata[test_dataset,]
test2 <- testdata[-test_dataset,]

#summrize result 

str(train_dataset) #structure of data
summary(train_dataset) #sumrry for data
nrow(train_dataset) 
prop.table(table(train_dataset$lable)) # lable probability 

########################################

str(test1) #structure of data
summary(test1) #sumrry for data
nrow(test1) 
prop.table(table(test1$lable)) # lable probability 

##########################################


str(test2) #structure of data
summary(test2) #sumrry for data
nrow(test2) 
prop.table(table(test2$lable)) # lable probability 

#------------------------ data preprocessing ----------------------------------------#

train_corpus <- Corpus(VectorSource(train_dataset$content))
length(train_corpus)


train_corpus[["1"]][["content"]]
train_corpus[["2"]][["content"]]
train_corpus[["3"]][["content"]]
train_corpus[["4"]][["content"]]
#########################################step1

step1 <-tm_map(train_corpus,tolower)#put all char in corpus as lowercase

step1[["1"]][["content"]]
step1[["2"]][["content"]]
step1[["3"]][["content"]]
step1[["4"]][["content"]]
inspect(head(step1,5))

##################################################step2

step2 <-tm_map(step1,removeNumbers)
step2[["1"]][["content"]]
inspect(step2[1:2])
inspect(head(step2,5))

#############################################################step3

step3 <-tm_map(step2,removeWords,stopwords("english"))

step3[["1"]][["content"]]
inspect(step3[1:2])
inspect(head(step3,5))

stopwordlist<- c(stopwords("english"),"one","two" ,"done","now", "tree","four","five","six","sven","eight","nine","teen", "movie","movies","films","film","song","songs","make","br")

step3.1 <-tm_map(step3,removeWords,stopwordlist)
inspect(step3.1[1:2])
inspect(head(step3.1,10))


############################################################## step4
step4 <- tm_map(step3.1,removePunctuation)
inspect(head(step4,10))

##################################################### step5
step5 <- tm_map(step4,stripWhitespace)
inspect(head(step5,10))

#--------------------------------------stemming-------------------------------------

#test stemmer 

testcase <- c("Do you really think it is weakness that yields to temptation?"," I tell you that there are terrible 
temptations which ","it requires strength"," strength and courage to yield to ~ Oscar Wilde")

corpus1 <- Corpus(VectorSource(testcase))
test_result<-tm_map(corpus1,stemDocument,language = "english")


step6<-tm_map(step5,stemDocument,language = "english")
#step6.1 <-tm_map(step6,stemCompletion(step6,dictionary = step5))
inspect(head(step6,10))

#------------------------------------ document term matrex ----------------------------

traing_dtm = DocumentTermMatrix(step6)
train_dtm <-DocumentTermMatrix(train_corpus, control = list(tolower = T , removeNumbers =T ,removePunctuation = T , stopwords = T 
                                                           , stripWhitespace = T   ))
traing_dtm <- as.matrix(train_dtm)
dim(train_dtm)

train_dtm_srt <- removeSparseTerms(train_dtm,0.99) #dimensionality reduction
dim(train_dtm_srt)

########################################################################### avrage frequency of most frequent words
mean_train =sort(colMeans(as.matrix(train_dtm_srt)),decreasing = T)
mean_train[1:20] # top 20  most frequent words
mean_train[1:40] # top 40  most frequent words
mean_train[1:10] # top 10  most frequent words
##########################################################################
average_top20=mean(mean_train[1:20]) # the average frequency of these word 
average_top20
################## plot data
barplot(mean_train[1:20],border = NA , las =3 ,xlab = "top 20 word " , ylab = "frequency" , ylim = c(0,1.5))

wordcloud( names(mean_train[1:40]),mean_train[1:40],scale = c(3,1) ,colors = brewer.pal(8,"Dark2"))


#----------------------------------------------------------------------------------- train data model

train_matrix <- as.matrix(train_dtm_srt)
train_data_model <- data.frame(y=train_dataset$lable , x = train_matrix)
### summrize
str(train_data_model)
prop.table(table(train_data_model$y))
nrow(train_data_model)
summary(train_data_model)

####################################################### save bag of word 

train_bag_of_word <- findFreqTerms(train_dtm_srt)

length(train_bag_of_word)

############################ generate test1_data_model &test2_data_model  

test1_corpus <- Corpus(VectorSource(as.matrix(test1$content)))
test1_corpus[["1"]][["content"]]
test1_corpus[["2"]][["content"]]
test1_corpus[["3"]][["content"]]

############################################ test1 as document term matrix 
# stemming = T 
test1_dtm <-DocumentTermMatrix(test1_corpus, control = list(tolower = T , removeNumbers =T ,removePunctuation = T , stopwords = T 
                                                           , stripWhitespace = T  ,dictionary = train_bag_of_word))

str(test1_dtm)
dim(test1_dtm)
### test1 matrix form 
test1_matrix <- as.matrix(test1_dtm)

##test1_data_model

test1_data_model <-  data.frame(y=test1$lable , x = test1_matrix)

summary(test1_data_model)

######################################################################################### test2 

test2_corpus <- Corpus(VectorSource(as.matrix(test2$content)))
test2_corpus[["1"]][["content"]]
test2_corpus[["2"]][["content"]]
test2_corpus[["3"]][["content"]]

############################################ test2 as document term matrix 

test2_dtm <-DocumentTermMatrix(test2_corpus, control = list(tolower = T , removeNumbers =T ,removePunctuation = T , stopwords = T 
                                                            , stripWhitespace = T ,dictionary = train_bag_of_word))

str(test2_dtm)
dim(test2_dtm)
### test1 matrix form 
test2_matrix <- as.matrix(test2_dtm)

##test1_data_model

test2_data_model <-  data.frame(y=test2$lable , x = test2_matrix)

summary(test2_data_model)

#-------------------------------------------------------------------------------------------------
# visualization


library('Rtsne')
library('ggplot2')
library('plotly')
library('tsne')


features <- stm_train_data_model[, !names(train_data_model) %in% c("y")]


tsne <- Rtsne(
  as.matrix(features),
  check_duplicates = FALSE,
  perplexity = 30,
  theta = 0.5,
  dims = 2,
  verbose = TRUE
)

embedding <- as.data.frame(tsne$Y)
embedding$Class <- as.factor(train_data_model$y)


ax <- list(title = ""
           ,zeroline = FALSE)

p <- plot_ly(
  data = embedding
  ,x = embedding$V1
  ,y = embedding$V2
  ,color = embedding$Class
  ,type = "scattergl"
  ,mode = 'markers'
  ,marker = list(line = list(width = 2))
  ,colors = c("#FF0000FF", "#CCFF00FF")
  ,symbols = c( "square", "triangle-down")
) %>% layout(xaxis = ax, yaxis = ax)
p

#---------------------------------------------- classification -----------------------------------

library(party)


#************************************* DICISION TREE *******************************
dtree_model <- ctree(y ~ . ,data = train_data_model)
summary(dtree_model)
plot(dtree_model)
plot(dtree_model,type = "simple")

########################## prediction process 


test1pred = predict(dtree_model,newdata = test1_data_model )

summary(test1pred)
prop.table(table(test1pred))

############################################## testing and evaluate the prediction

#first confusion matrix 

confusionMatrix(test1pred,test1_data_model[,1],positive = "positive", dnn = c ("prediction","true"))


# second accuracy

mmetric(test1pred,test1_data_model[,1],c("ACC","TPR","PRECISION","F1"))

#*************************************** NAIVE BAYES CLASSIFIER ***************************

library(e1071)

naiveB_model = naiveBayes(y~.,data =train_data_model )

########################## prediction process 

naiveB_test1pred = predict(naiveB_model,newdata = test1_data_model )

summary(naiveB_test1pred)
prop.table(table(naiveB_test1pred))

############################################## testing and evaluate the prediction

#first confusion matrix 

confusionMatrix(naiveB_test1pred,test1_data_model[,1],positive = "positive", dnn = c ("prediction","true"))


# second accuracy

mmetric(naiveB_test1pred,test1_data_model[,1],c("ACC","TPR","PRECISION","F1"))

#***************************** k-nearest neighbour classification *************************************************

library(class)

knn_model = knn(train = train_data_model[,-1] , test = test1_data_model[,-1] , cl = train_data_model[,1] ,k = 2 )

summary(knn_model)
prop.table(table(knn_model))

############################################## testing and evaluate the prediction

#first confusion matrix 

confusionMatrix(knn_model,test1_data_model[,1],positive = "positive", dnn = c ("prediction","true"))

# second accuracy

mmetric(knn_model,test1_data_model[,1],c("ACC","TPR","PRECISION","F1"))
#*********************************** SVM **************************************
library(caret)

svm_classifier = svm(formula = y ~ . , data = train_data_model , 
                type = "C-classification" , kernel = 'radial')



########################## prediction process 

SVM_test1pred = predict(svm_classifier,newdata = test1_data_model )
summary(SVM_test1pred)
prop.table(table(SVM_test1pred))

############################################## testing and evaluate the prediction

#first confusion matrix 

confusionMatrix(SVM_test1pred,test1_data_model[,1],positive = "positive", dnn = c ("prediction","true"))


# second accuracy

mmetric(SVM_test1pred,test1_data_model[,1],c("ACC","TPR","PRECISION","F1"))






#***************************** kernal SVM **************************************

library(kernlab)

KSVM_model = ksvm(y~ . , data =train_data_model)


########################## prediction process 

test2_data_model$x <-0

KSVM_test1pred = predict(KSVM_model,newdata = test1_data_model )
KSVM_test2pred = predict(KSVM_model,newdata = test2_data_model )
summary(KSVM_test1pred)
summary(KSVM_test2pred)

prop.table(table(KSVM_test1pred))

############################################## testing and evaluate the prediction

#first confusion matrix 

confusionMatrix(KSVM_test1pred,test1_data_model[,1],positive = "positive", dnn = c ("prediction","true"))

confusionMatrix(KSVM_test2pred,test2_data_model[,1],positive = "positive", dnn = c ("prediction","true"))


# second accuracy

mmetric(KSVM_test1pred,test1_data_model[,1],c("ACC","TPR","PRECISION","F1"))

mmetric(KSVM_test2pred,test2_data_model[,1],c("ACC","TPR","PRECISION","F1"))


#******************************************** random forest classifier ******************************************
install.packages("ggplot2")
install.packages("randomForest")
library(randomForest)

rf_model <- randomForest(x = train_data_model[,-1],y = train_data_model$y , ntree = 40)


########################## prediction process 

rf_test1pred = predict(rf_model,newdata = test1_data_model )
summary(rf_test1pred)
prop.table(table(rf_test1pred))

############################################## testing and evaluate the prediction

#first confusion matrix 

confusionMatrix(rf_test1pred,test1_data_model[,1],positive = "positive", dnn = c ("prediction","true"))


# second accuracy

mmetric(rf_test1pred,test1_data_model[,1],c("ACC","TPR","PRECISION","F1"))




#=============================================================================================================

#==============================================================================================================

#stamming


stem_dtm <- function(dtm)
{
  n_char = 1
  stem <- function(word)
  {
    common_endings <- c("*ed$", "*ing$", "*s$", "*es$", 
                        "*ly$", "*ary$", "*self$", "*ful$", 
                        "*less$","*ment$", "*er$", "*ance$",
                        "*al$", "*ent$", "*sion$", "*tion$",
                        "*ance$", "*or$", "*ive$", "*ise$")
    # remove common endings
    for(i in 1:length(common_endings)){word <- sub(common_endings[i], "", word)}
    return(word)
  }
  predictors <- colnames(dtm)
  stemmed_predictors <- stem(colnames(dtm))
  duplicated_terms <- stemmed_predictors[duplicated(stemmed_predictors, 
                                                    incomparables = FALSE)]
  duplicated_terms <- unique(duplicated_terms[nchar(duplicated_terms) > n_char])
  stemmed_dtm <- matrix(NA, 
                        nrow = nrow(dtm), 
                        ncol=length(duplicated_terms))
  for(i in 1:length(duplicated_terms))
  {
    duplicated_columns <- grep(duplicated_terms[i], predictors)
    replacement_column <- rowSums(dtm[,duplicated_columns])
    stemmed_dtm[,i] <- replacement_column
  }
  print("Done")
  colnames(stemmed_dtm) <- duplicated_terms
  #stemmed_dtm <- (stemmed_dtm > 0)+0
  return(stemmed_dtm)
}

stm_train_matrix <- as.matrix(train_dtm)
stm_train_data_model <- as.data.frame(stm_train_matrix)


stmQ <- as.DocumentTermMatrix(stm_train_data_model ,weighting = 1)
stm_train_bag_of_word <- findFreqTerms(stmQ)
length(stm_train_bag_of_word)

stm_mean_train =sort(colMeans(as.matrix(stm)),decreasing = T)
barplot(stm_mean_train[1:60],border = NA , las =3 ,  ylab = "frequency" ,space = F)



stm_test1_dtm <-DocumentTermMatrix(test1_corpus,
                                   control = list(tolower = T 
                                                  , removeNumbers =T ,removePunctuation = T 
                                                  , stopwords = T , stripWhitespace = T ,
                                                  dictionary = stm_train_bag_of_word )
)
dim(stm_test1_dtm)
### test1 matrix form 
stm_test1_matrix <- as.matrix(stm_test1_dtm)
stm_test1_data_model <- as.data.frame(stm_test1_matrix)
##test1_data_model

stm <- stem_dtm(stm_train_data_model)
stm1 <-stem_dtm(stm_test1_data_model) 



stm_train_data_model <- data.frame(y=train_dataset$lable , x = stm)
stm_test1_data_model <-  data.frame(y=test1$lable , x =stm1)

dim(stm_train_data_model)
dim(stm_test1_data_model)

#save(stm_train_data_model,file = "train/stm_train_data_model.RData")
#save(stm_test1_data_model,file = "train/stm_test1_data_model.RData")


load("train/stm_train_data_model.RData")
load("train/stm_test1_data_model.RData")

stm_KSVM_model = ksvm( stm_train_data_model$y~ . , data =stm_train_data_model)
########################## prediction process 
stm_KSVM_test1pred = predict(stm_KSVM_model,newdata = stm_test1_data_model)
############################################## testing and evaluate the prediction
#first confusion matrix 
#confusionMatrix(stm_KSVM_test1pred,stm_test1_data_model[,1],positive = "positive", dnn = c ("prediction","true"))
# second accuracy
table(stm_KSVM_test1pred,stm_test1_data_model[,1])
mmetric(stm_KSVM_test1pred,stm_test1_data_model[,1],c("ACC","TPR","PRECISION","F1"))
#-------------------------------------------------------------------------------------------------------
library(randomForest)

rf_model <- randomForest(x = stm_train_data_model[,-1],y = stm_train_data_model$y , ntree = 60)


########################## prediction process 

rf_test1pred = predict(rf_model,newdata = stm_test1_data_model )
summary(rf_test1pred)
prop.table(table(rf_test1pred))

############################################## testing and evaluate the prediction

#first confusion matrix 

confusionMatrix(rf_test1pred,stm_test1_data_model[,1],positive = "positive", dnn = c ("prediction","true"))


# second accuracy

mmetric(rf_test1pred,stm_test1_data_model[,1],c("ACC","TPR","PRECISION","F1"))


