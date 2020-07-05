# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python37_app]
from flask import Flask, request, jsonify

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
from collections import Counter

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)


@app.route('/getsign', methods = ['POST'])
def hello():
    """Return a friendly HTTP greeting."""
    dataJson = request.get_json(force=True)
    final_predictions = dict()
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

    scaler = joblib.load('min_max_scaler.sav')
    X_after = scaler.transform(np.array(test_data))
    X_test = X_after[:, 0:38]

    knn = joblib.load('knn.sav')
    decisionTree = joblib.load('decision_tree.sav')
    randomForest = joblib.load('random_forest.sav')
    neuralNetwork = joblib.load('neural_network.sav')

    final_predictions[1] = Counter(knn.predict(X_test).tolist()).most_common(1)[0][0]
    final_predictions[2] = Counter(decisionTree.predict(X_test)).most_common(1)[0][0]
    final_predictions[3] = Counter(randomForest.predict(X_test)).most_common(1)[0][0]
    final_predictions[4] = Counter(neuralNetwork.predict(X_test)).most_common(1)[0][0]

    return jsonify(final_predictions)


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python37_app]
