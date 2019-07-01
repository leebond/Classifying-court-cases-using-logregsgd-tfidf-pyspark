## Repository description
In this repository, I am attempting the use of pyspark to build a logistic regression model with SGD. In the second script, I will evaluate the model using the tested data.

## The commands below will not run on your local machine with spark being setup.Make sure you have cloud computing resources eg. EMR or GCP to run the below jobs.

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
