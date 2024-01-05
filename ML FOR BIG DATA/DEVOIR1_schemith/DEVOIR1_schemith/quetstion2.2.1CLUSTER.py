# -*- coding: utf-8 -*-
import time
from pyspark.sql import SparkSession

try:
    sc.stop()
except:
    print()

spark = SparkSession.builder\
        .master("local[8]")\
        .appName("Abdenour")\
        .getOrCreate()

from pyspark import SparkContext
sc = spark.sparkContext.getOrCreate()

from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.util import MLUtils
result_file = open("resultats.txt", 'w') 
input_str = "Starting Experiment..." + '\n'
data = MLUtils.loadLibSVMFile(sc, 'iris_libsvm_large.txt')
for i in range (1,11):
    start_time = time.time()
    model = DecisionTree.trainClassifier(data, numClasses=3, categoricalFeaturesInfo={}, impurity='entropy', maxDepth=5, maxBins=32)
    end_time = time.time()
    elapsed_time = end_time - start_time
    input_str += "Elapsed time:" +  str(elapsed_time) +"seconds for  time the data sample\n\n"
    data = data.union(data)
result_file.write(input_str)
result_file.close()
