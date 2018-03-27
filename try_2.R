library(tidytext)
library(tm)
library(caret)
library(kernlab)
library(e1071)
library(jsonlite)
library(stringr)
setwd("C:/Users/Jordan/Desktop/DS340")
run_svm = function(){
  ######################LOADING IN DATA #####################################
  ##we'll only use small data for now, as R can't handle the bigger data locally - that's an XSEDE problem for later this week
  x = stream_in(file("big/instances.jsonl",open="r"))
  y = stream_in(file("big/truth.jsonl",open="r"))
  train = merge(x,y,by="id")
  train = train[1:12000,]
  train$clickbait = as.factor(ifelse(train$truthClass=="clickbait","clickbait","good"))
  
  f = function(x){paste(unlist(x),collapse= " ")}
  train$targetParagraphs = sapply(train$targetParagraphs,f)
  train$targetCaptions = sapply(train$targetCaptions,f)
  postText = train[,c("postText","targetTitle","clickbait")]
  postText = as.data.frame(lapply(postText, function(x){as.character(str_replace_all(x,"[^[:graph:]]", " "))}))
  new = postText[,c(ncol(postText),1:(ncol(postText)-1))]
  new = as.data.frame(lapply(new,as.character))
  new$clickbait = as.factor(new$clickbait)
  ##########################################################################
  
  ###paste(unlist(v),collapse=" ") works on a single element of train$targetParagraphs
  
  
  ############CLEANING############################################
  ##removing weird characters, whitespace, punctuation, stopwords, and making everything lowercase.
  ##leaving numbers, as those are likely important in clickbait
  
  new2 = lapply(new[,-1],function(x){Corpus(VectorSource(x))})
  new2 = lapply(new2,function(x){tm_map(x,function(y){iconv(enc2utf8(y),sub="byte")})})
  new2 = lapply(new2,function(x){tm_map(x,content_transformer(tolower))})
  new2 = lapply(new2,function(x){tm_map(x, removeWords, stopwords())})
  new2 = lapply(new2,function(x){tm_map(x, stripWhitespace)})
  new2 = lapply(new2,function(x){tm_map(x, removePunctuation)})
  
  ##Creating Document Term Matrices with tm as part of cleaning
  dtm = lapply(new2,DocumentTermMatrix)
  features = lapply(dtm,function(x){findFreqTerms(x,10)})
  dtm2 = list()
  for(i in seq(1,length(features))){
    dtm_i = DocumentTermMatrix(new2[[i]], list(global = c(2, Inf),
                                         dictionary = features[[i]]))
    dtm2[[i]] = dtm_i
    
  }
  
  #####dividing training and testing data ####
  ##when ran on supercomputer later, will be able to handle more training/testing data
  #train1 = new[1:(dim(new)[1]-100),]   ##for now, taking the last 100 records of our dataset to be our testing data.  When we run this on XSEDE, we'll have two separate sets for training and testing
  #test1 = new[(dim(new)[1]-99):dim(new)[1],]
  set.seed(440)
  trainIndex = createDataPartition(new$clickbait,p=.8,list=FALSE)
  train1 = new[trainIndex,]
  test1 = new[-trainIndex,]
  #train2 = new2[trainIndex,]
  train2 = lapply(new2,function(x){x[trainIndex]})
  test2 = lapply(new2,function(x){x[-trainIndex]})
  
  #######DTM --> Data Frame######################
  dict2 = lapply(dtm2,function(x){findFreqTerms(x,lowfre=10)})
  training_clickbait = list()
  testing_clickbait = list()
  for(i in seq(1,length(train2))){
    training_clickbait[[i]] = DocumentTermMatrix(train2[[i]],list(dictionary=dict2[[i]]))
    testing_clickbait[[i]] = DocumentTermMatrix(test2[[i]],list(dictionary=dict2[[i]]))
    
  }
  
  ##allows for DTM --> data frame
  training_clickbait = as.data.frame(lapply(training_clickbait,function(x){as.data.frame(apply(x,MARGIN=2,FUN=function(y){y=ifelse(y>0,1,0)}))}))
  testing_clickbait = as.data.frame(lapply(testing_clickbait,function(x){as.data.frame(apply(x,MARGIN=2,FUN=function(y){y=ifelse(y>0,1,0)}))}))
  
  #########preparing for svm function#####################
  
  training_clickbait <- cbind(clickbait=factor(train1$clickbait), training_clickbait)
  testing_clickbait <- cbind(clickbait=factor(test1$clickbait), testing_clickbait)
  print("Made it to the model")
  model <- svm(clickbait ~., data=training_clickbait)
  #print(model)
  
  prediction <- predict(model, testing_clickbait)
  
  comparison <- table(testing_clickbait$clickbait, prediction, dnn=c("Actual", "Predicted"))
  print(comparison)
  
  ###########################
  
  visual <- confusionMatrix(prediction, testing_clickbait$clickbait, positive="clickbait")
  cat(visual$overall[1], "default", "\n", sep = ",")

  return("Sahhhhhhhh")
}