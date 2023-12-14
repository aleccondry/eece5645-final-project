import numpy as np
import pandas as pd
import argparse
import findspark
import random
from time import time
from pyspark.sql.types import StructType, StructField, FloatType, ArrayType
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint




def read_Data(input_file):
    
    l = len(input_file)
    features = np.array(input_file)
    data = []
    for value in range(l):
        data.append((features[value,0],(features[value,1:l])))
    
    return data


def create_data_rdd(data,N):
    
    data_rdd = sc.parallelize([(float(label), features.tolist()) for label, features in data],N) .map(lambda x: LabeledPoint(x[0], x[1]))
    return data_rdd


def create_data_txt(data):

    schema = StructType([StructField("label", FloatType(), True), StructField("features", ArrayType(FloatType(), containsNull=False), True)])
    data_df = spark.createDataFrame(data,schema=schema)
    data_rdd = data_df.rdd.map(lambda row: LabeledPoint(row["label"], row["features"]))
    data_rdd.saveAsTextFile("labelled_data.txt")


def metrics_stats(labelsAndPredictions,c):

    metrics = MulticlassMetrics(labelsAndPredictions)
    Accuracy = labelsAndPredictions.filter(lambda lp: lp[0] == lp[1]).count() / float(c)
    precision = metrics.precision()
    recall = metrics.recall()
    return Accuracy, precision, recall



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Random Forest Classifier in Parallel',formatter_class=argparse.ArgumentDefaultsHelpFormatter)     
    parser.add_argument('--input',default="/home/ghosh.arp/Project/Training_data.csv", help='csv file to be processed')
    parser.add_argument('--output',default="/home/ghosh.arp/Project/Test_data.csv", help='csv file to be processed')
    parser.add_argument('--master',default="local[25]",help="Spark Master")
    parser.add_argument('--N',type=int,default=20,help="Number of partitions to be used in RDDs.")
    parser.add_argument('--ratio',type=float,default=0.7,help="Ratio for partitioning the data into Training set and Test set")
    parser.add_argument('--tree',type=int,default=20, help="Number of trees for Random Forest Classifier")
    args = parser.parse_args()

    sc = SparkContext(args.master, 'RandomForest')
    sc.setLogLevel('warn')

    input_file =  pd.read_csv(args.input)
    output_file =  pd.read_csv(args.output)
    new_partition = args.N
    r = float(args.ratio)
   
    input_rdd = create_data_rdd(read_Data(input_file),args.N)
    output_rdd = create_data_rdd(read_Data(output_file),args.N)
    
    (trainingData, testData) = input_rdd.randomSplit([r, (1-r)])
    
    start = time()
    model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, numTrees=args.tree, featureSubsetStrategy="auto",
                impurity='gini', maxDepth=4, maxBins=32)
    end = time()
    print("Training Time:", str(end-start))

    start = time()
    predictions = model.predict(testData.map(lambda x: x.features))
    end = time()
    print("Testing Time:", str(end-start))

    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    
    print("The metrics on the test data of the training set")
    start = time()
    print(metrics_stats(labelsAndPredictions,testData.count()))
    end = time()
    print("Metrics Calculation Time:", str(end-start))
    
    start = time()
    predictions_test = model.predict(output_rdd.map(lambda x: x.features))
    end = time()
    print("Prediction time on new data:", str(end-start))

     
    labelsAndPredictions_test = output_rdd.map(lambda lp: lp.label).zip(predictions_test)

    print("The metrics on the Output Test data")
    
    start = time()
    print(metrics_stats(labelsAndPredictions_test,output_rdd.count()))
    end =time()
    print("Metrics Calculation Time:", str(end-start))





