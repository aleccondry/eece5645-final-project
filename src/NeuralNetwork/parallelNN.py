from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession.builder.appName("DiabetesPrediction").getOrCreate()


data = spark.read.csv('../../data/Clean_data.csv', header=True, inferSchema=True)


data = data.withColumnRenamed('Diabetes_binary', 'label')  

data = data.withColumn('label', when(col('label') == 1.0, 1).otherwise(0).cast('int'))
assembler = VectorAssembler(
    inputCols=['HighBP','HighChol','CholCheck','BMI','Smoker',
               'Stroke','HeartDiseaseorAttack','PhysActivity',
               'Fruits','Veggies','HvyAlcoholConsump',
               'AnyHealthcare','NoDocbcCost','GenHlth',
               'MentHlth','PhysHlth','DiffWalk','Sex',
               'Age','Education','Income'
],
    outputCol='features'
)

data = assembler.transform(data)


train_data, test_data = data.randomSplit([0.75, 0.25], seed=1)

layers = [21, 8, 8, 21]
mlp = MultilayerPerceptronClassifier(layers=layers,  labelCol='label', seed=1) 

model = mlp.fit(train_data)

predictions = model.transform(test_data)
predictions.show()

evaluator = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")


predictions.select("prediction", "label").show(3, truncate=False)


spark.stop()
