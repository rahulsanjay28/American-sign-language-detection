import json
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import pandas as pd

count = 0
path = "/Users/rahulsanjay/Downloads/cse535/Thursday_Assignment_2_json/"
parts = ['lsh', 'rsh', 'lelb', 'relb', 'lw', 'rw']
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith(".json"):
            with open(os.path.join(root, f)) as json_file:
                dataJson = json.load(json_file)
                for image in dataJson:
                    count = count + 1
                    xcor = []
                    ycor = []
                    points = []
                    # for keypoint in image['keypoints'][5:11]:
                    #     # print(keypoint['part'], keypoint['position']['x'])
                    #     temp = []
                    #     temp.append(keypoint['position']['x'])
                    #     temp.append(keypoint['position']['y'])
                    #     points.append(temp)
                    # scaler = preprocessing.MinMaxScaler()
                    # X_after = scaler.fit_transform(np.array(points))
                    # train_data = X_after.flatten().tolist()
                    # if "car" in root:
                    #     train_data.append('car')
                    # elif "book" in root:
                    #     train_data.append('book')
                    # elif "sell" in root:
                    #     train_data.append('sell')
                    # elif "movie" in root:
                    #     train_data.append('movie')
                    # elif "gift" in root:
                    #     train_data.append('gift')
                    # elif "total" in root:
                    #     train_data.append('total')


                    # for keypoint in image['keypoints'][5:11]:
                    #     points.append(keypoint['position']['x'])
                    #     points.append(keypoint['position']['y'])
                    # print("------------------------------------------")

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
                        if(keypoint['part'] != "leftElbow" and keypoint['part'] != "rightElbow"):
                            temp = np.array([keypoint['position']['x'], keypoint['position']['y']])
                            points.append(np.linalg.norm(temp - lelbow))
                            points.append(np.linalg.norm(temp - relbow))

                    # print(root)
                    if "car" in root:
                        points.append('car')
                    elif "book" in root:
                        points.append('book')
                    elif "sell" in root:
                        points.append('sell')
                    elif "movie" in root:
                        points.append('movie')
                    elif "gift" in root:
                        points.append('gift')
                    elif "total" in root:
                        points.append('total')

                    with open(path + "key_points.csv", 'a') as f:
                        pd.DataFrame([points]).to_csv(f, index=False, header=False)

                    # if count == 7000:
                    #     for i in range(len(X_after)):
                    #         plt.plot(X_after[i][0], X_after[i][1], marker='o', markersize=3, color="red", label='point')
                    #         plt.text(X_after[i][0]+0.01, X_after[i][1] + 0.01, parts[i], fontsize=9)
                    #     plt.gca().invert_yaxis()
                    #     plt.show()