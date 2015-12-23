Practical Machine Learning - Course Project - Writeup
=====================================================

In this assignment, the provided data have been analyzed to determine
what activity an individual can perform. In order to do this, package
caret and randomForest have been used. This will allow generating
correct answers in this assignment. A seed value have been used to get
consistent results.

    library(Hmisc)

    ## Loading required package: grid
    ## Loading required package: lattice
    ## Loading required package: survival
    ## Loading required package: Formula
    ## Loading required package: ggplot2
    ## 
    ## Attaching package: 'Hmisc'
    ## 
    ## The following objects are masked from 'package:base':
    ## 
    ##     format.pval, round.POSIXt, trunc.POSIXt, units

    library(caret)

    ## 
    ## Attaching package: 'caret'
    ## 
    ## The following object is masked from 'package:survival':
    ## 
    ##     cluster

    library(randomForest)

    ## randomForest 4.6-12
    ## Type rfNews() to see new features/changes/bug fixes.
    ## 
    ## Attaching package: 'randomForest'
    ## 
    ## The following object is masked from 'package:Hmisc':
    ## 
    ##     combine

    library(foreach)
    library(doParallel)

    ## Loading required package: iterators
    ## Loading required package: parallel

    set.seed(12345)
    options(warn=-1)

Data provided have been loaded that are training and test data. Values
contained "\#DIV/0!" have been replaced by an NA value.

    training_data <- read.csv("pml-training.csv", na.strings=c("#DIV/0!") )
    testing_data <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!") )

All columns 8 to the end have been casted to numeric.

    for(i in c(8:ncol(training_data)-1)) {training_data[,i] = as.numeric(as.character(training_data[,i]))}

    for(i in c(8:ncol(testing_data)-1)) {testing_data[,i] = as.numeric(as.character(testing_data[,i]))}

Some columns were mostly blank. Hence, good prediction cannot be
achieved. Feature sets that include complete column only will be chosen.
Username, timestamps and windows also have been removed. Determination
of feature sets have been done and displayed.

    feature_set <- colnames(training_data[colSums(is.na(training_data)) == 0])[-(1:7)]
    model_data <- training_data[feature_set]
    feature_set

    ##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
    ##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
    ##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
    ## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
    ## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
    ## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
    ## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
    ## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
    ## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
    ## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
    ## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
    ## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
    ## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
    ## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
    ## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
    ## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
    ## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
    ## [52] "magnet_forearm_z"     "classe"

Model data built from selected feature set have been obtained.

    index <- createDataPartition(y=model_data$classe, p=0.75, list=FALSE )
    training <- model_data[index,]
    testing <- model_data[-index,]

Five random forests with 150 trees each have been developed. Parallel
processing have been used to build this model.

    registerDoParallel()
    x <- training[-ncol(training)]
    y <- training$classe

    rf <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
    randomForest(x, y, ntree=ntree) 
    }

Error reports for both training and test data have been generated.

    predictions1 <- predict(rf, newdata=training)
    confusionMatrix(predictions1,training$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 4185    0    0    0    0
    ##          B    0 2848    0    0    0
    ##          C    0    0 2567    0    0
    ##          D    0    0    0 2412    0
    ##          E    0    0    0    0 2706
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9997, 1)
    ##     No Information Rate : 0.2843     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
    ## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
    ## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

    predictions2 <- predict(rf, newdata=testing)
    confusionMatrix(predictions2,testing$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1395    7    0    0    0
    ##          B    0  939    2    0    0
    ##          C    0    3  851    6    1
    ##          D    0    0    2  798    4
    ##          E    0    0    0    0  896
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9949          
    ##                  95% CI : (0.9925, 0.9967)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9936          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9895   0.9953   0.9925   0.9945
    ## Specificity            0.9980   0.9995   0.9975   0.9985   1.0000
    ## Pos Pred Value         0.9950   0.9979   0.9884   0.9925   1.0000
    ## Neg Pred Value         1.0000   0.9975   0.9990   0.9985   0.9988
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2845   0.1915   0.1735   0.1627   0.1827
    ## Detection Prevalence   0.2859   0.1919   0.1756   0.1639   0.1827
    ## Balanced Accuracy      0.9990   0.9945   0.9964   0.9955   0.9972

Conclusions and Test Data Submit
--------------------------------

Confusion matrix obtained from this model is very accurate. Because test
data was around 99% accurate, it is expected nearly all of the submitted
test cases to be correct. It turned out they were all correct.

Submission for this course project have been prepared by COURSERA.

    pml_write_files = function(x){
      n = length(x)
      for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
      }
    }


    x <- testing_data
    x <- x[feature_set[feature_set!='classe']]
    answers <- predict(rf, newdata=x)

    answers

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

    pml_write_files(answers)
