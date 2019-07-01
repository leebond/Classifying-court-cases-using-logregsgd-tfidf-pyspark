# Repository description
In this repository, I am attempting the use of pyspark to build a logistic regression model with SGD. In the second script, I will evaluate the model using the tested data.

### Business case
We want to know if a document belongs to an Australian court case or not.

### Datasets
Full sized:
The train data consists of 170,000 text documents of 7.6 million lines of text (1.9GB).
The test data consists of 18.700 text documents of 1 million lines of text (200 MB).

Small sized:
The small train data is about 37.5MB.

I have tested my code on the small train data to build my logistic regression model. Once my code is ready, I will upload the scripts to AWS or GCP to train on the full dataset and also test on the full test dataset.


For GCP:
Full sized training data is located at gs://metcs777/TrainingData.txt
Full sized testing data is located at gs://metcs777/TestingData.txt
Small sized training data is located at gs://metcs777/SmallTrainingData.txt

For AWS:
Full sized training data is located at s3://metcs777/TrainingData.txt
Full sized testing data is located at s3://metcs777/TestingData.txt
Small sized training data is located at s3://metcs777/SmallTrainingData.txt


## Local Commands
The commands below will not run on your local machine with spark being setup.Make sure you have cloud computing resources eg. EMR or GCP to run the below jobs.

### Train
format:
spark-submit <train.py> <training data> <outputdirectory>

example:
spark-submit logistic_regression_pyspark_train.py  gs://metcs777/TrainingData.txt outputdir 

### Test
format:
spark-submit <test.py> <test data> <trained_model_directory>

example:
spark-submit prediction.py gs://metcs777/TestingData.txt
