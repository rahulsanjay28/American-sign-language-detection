import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import numpy as np
import pandas as pd
import statistics
import json
import joblib
import os
from collections import Counter

scaler = joblib.load('min_max_scaler.sav')
knn = joblib.load('knn.sav')
decisionTree = joblib.load('decision_tree.sav')
randomForest = joblib.load('random_forest.sav')
neuralNetwork = joblib.load('neural_network.sav')
# print(knn)
final_predictions = dict()

totalBookCount = 0
totalSellCount = 0
totalTotalCount = 0
totalGiftCount = 0
totalMovieCount = 0
totalCarCount = 0

totalCount = 0
knnCount = 0
decisionTreeCount = 0
randomForestCount = 0
neuralNetworkCount = 0

path = "/Users/rahulsanjay/Downloads/cse535/test_videos/"
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith(".json"):
            with open(os.path.join(root, f)) as json_file:
                dataJson = json.load(json_file)
                test_data = []
                for image in dataJson:
                    points = []
                    # for keypoint in image['keypoints'][5:11]:
                    #     points.append(keypoint['position']['x'])
                    #     points.append(keypoint['position']['y'])

                    lw = np.array([image['keypoints'][9]['position']['x'], image['keypoints'][9]['position']['y']])
                    rw = np.array([image['keypoints'][10]['position']['x'], image['keypoints'][10]['position']['y']])
                    points.append(np.linalg.norm(lw - rw))
                    for keypoint in image['keypoints'][0:9]:
                        temp = np.array([keypoint['position']['x'], keypoint['position']['y']])
                        points.append(np.linalg.norm(temp - lw))
                        points.append(np.linalg.norm(temp - rw))

                    lelbow = np.array([image['keypoints'][7]['position']['x'], image['keypoints'][7]['position']['y']])
                    relbow = np.array([image['keypoints'][8]['position']['x'], image['keypoints'][8]['position']['y']])
                    points.append(np.linalg.norm(lelbow - relbow))

                    for keypoint in image['keypoints'][0:11]:
                        if (keypoint['part'] != "leftElbow" and keypoint['part'] != "rightElbow"):
                            temp = np.array([keypoint['position']['x'], keypoint['position']['y']])
                            points.append(np.linalg.norm(temp - lelbow))
                            points.append(np.linalg.norm(temp - relbow))

                    test_data.append(points)

                X_after = scaler.transform(np.array(test_data))
                X_test = X_after[:, 0:38]

                predictionsList = knn.predict(X_test).tolist()
                # print("book", predictionsList.count('book'))
                # print("car", predictionsList.count('car'))
                # print("sell", predictionsList.count('sell'))
                # print("total", predictionsList.count('total'))
                # print("movie", predictionsList.count('movie'))
                # print("gift", predictionsList.count('gift'))
                final_predictions[1] = Counter(predictionsList).most_common(1)[0][0]

                predictionsList = decisionTree.predict(X_test).tolist()
                # print("book", predictionsList.count('book'))
                # print("car", predictionsList.count('car'))
                # print("sell", predictionsList.count('sell'))
                # print("total", predictionsList.count('total'))
                # print("movie", predictionsList.count('movie'))
                # print("gift", predictionsList.count('gift'))

                final_predictions[2] = Counter(predictionsList).most_common(1)[0][0]

                predictionsList = randomForest.predict(X_test).tolist()
                # print("book", predictionsList.count('book'))
                # print("car", predictionsList.count('car'))
                # print("sell", predictionsList.count('sell'))
                # print("total", predictionsList.count('total'))
                # print("movie", predictionsList.count('movie'))
                # print("gift", predictionsList.count('gift'))

                final_predictions[3] = Counter(predictionsList).most_common(1)[0][0]

                predictionsList = neuralNetwork.predict(X_test).tolist()
                # print("book", predictionsList.count('book'))
                # print("car", predictionsList.count('car'))
                # print("sell", predictionsList.count('sell'))
                # print("total", predictionsList.count('total'))
                # print("movie", predictionsList.count('movie'))
                # print("gift", predictionsList.count('gift'))
                final_predictions[4] = Counter(predictionsList).most_common(1)[0][0]

                # print(root.lower(), final_predictions)
                totalCount += 1
                if final_predictions[1] in root:
                    knnCount += 1
                if final_predictions[2] in root:
                    decisionTreeCount += 1
                if final_predictions[3] in root:
                    randomForestCount += 1
                if final_predictions[4] in root:
                    neuralNetworkCount += 1

print("knn ", knnCount/totalCount)
print("decisionTree ", decisionTreeCount/totalCount)
print("randomForest ", randomForestCount/totalCount)
print("neuralNetwork ", neuralNetworkCount/totalCount)