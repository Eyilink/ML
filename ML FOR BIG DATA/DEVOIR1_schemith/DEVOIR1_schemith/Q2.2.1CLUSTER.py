# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
import time


#Lecture du fichier
result_file = open("resultats.txt","w")

#Chaine de caractère à écrire
input_str =" Contenu de la page \n "
input_str = input_str + "ajout de contenu3"


try:
    sc.stop()
except:
    print("pas de sparkcontext")
    

spark = SparkSession.builder\
        .master("local[8]")\
        .appName("Ayoub")\
        .getOrCreate()


from pyspark import SparkContext
sc = spark.sparkContext.getOrCreate()

print(sc)

##Import des arbres de décision

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

print("import ok")
# Load and parse the data file into an RDD of LabeledPoint.
#data = MLUtils.loadLibSVMFile(sc, 'iris_libsvm.txt')


#Lecture des données 
data = MLUtils.loadLibSVMFile(sc, 'iris_libsvm.txt')
counter = 1 

data1 = data


#Séparation du training set
(trainingData, testData) = data.randomSplit([0.7, 0.3])

trainingData_c = trainingData
temp = trainingData
########## Pour 1 dataset calculons le temps 


input_str += "échantillon " + str(counter) #" b\n"

a = time.time()
model = DecisionTree.trainClassifier(trainingData, numClasses=3, categoricalFeaturesInfo={},impurity='entropy', maxDepth=5, maxBins=32)
b = time.time()
time_data = b-a 


input_str += "On mesure pour" + str(counter) + "fois le data set : " + str(time_data) + " \n"

# input_str += "on ajoute les données suivantes : " + model.toDebugString().encode('utf-8').decode('utf-8') #+ "b\n"
input_str += "on ajoute les données suivantes : " + str(model.toDebugString()) #+ "b\n"

#Prédiction du modèle 
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(testData.count())

input_str += "Test Error : " +  str(testErr) #+ "\n" 
print('Test Error = ' + str(testErr))
print('Learned classification tree model:')




###########Pour les suivants
for i in range(15):
    #On ajoute les données
    trainingData_c = trainingData_c.union(temp)
    input_str += "échantillon " + str(counter) #+ "\n"

    a = time.time()
    model = DecisionTree.trainClassifier(trainingData_c, numClasses=3, categoricalFeaturesInfo={},impurity='entropy', maxDepth=5, maxBins=32)
    b = time.time()
    time_data = b-a 


    input_str += "On mesure pour" + str(counter) + "fois le data set : " + str(time_data) #+ "\n"

    # input_str += "on ajoute les données suivantes : " + model.toDebugString() #+ "\n"
    input_str += "on ajoute les données suivantes : " + str(model.toDebugString()) #+ "b\n"


    #Prédiction du modèle 
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(
        lambda lp: lp[0] != lp[1]).count() / float(testData.count())

    input_str += "Test Error : " +  str(testErr) #+ "\n"
    print('Test Error = ' + str(testErr))
    print('Learned classification tree model:')


#for i in range(0): 
 #   data_temp = data1
  #  data = data.union(data_temp)
   # counter = counter +1

#result_data = data.count()

#print ("on a "  ,counter,"fois le dataset : ", result_data )
print ("on a "  ,counter,"fois le dataset : " )






#Partie écriture

result_file.write(input_str)
result_file.close()