from imageai import Prediction
from imageai.Prediction import ImagePrediction


import os
execution_path=os.getcwd()
Prediction=ImagePrediction()
Prediction.setModelTypeAsDenseNet()
Prediction.setModelPath(os.path.join(execution_path, "DenseNet-BC-121-32.h5"))
Prediction.loadModel()
# Prediction.setModelTypeAsSqueezeNet()
# Prediction.setModelPath(os.path)
predictions, probabilities = Prediction.predictImage(os.path.join(execution_path, "giraffe.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)