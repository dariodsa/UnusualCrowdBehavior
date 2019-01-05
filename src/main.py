#!/usr/bin/python

import matplotlib.pyplot as plt
import time
import sys
import os
import cv2
import numpy as np
import lda
import social_force as socialForce
from sklearn.cluster import KMeans

VIDEO_URL = "../datas/crowd-activity-all.avi"
NORMAL_DATASET = "../datas/normalScene2"

def kmeans(clusterSize, visualWords, n=5 , T=10):
    points = np.empty((0,n*n*T*3))
    print(len(visualWords))
    for word in visualWords:
        print("word", word)
        print(word.shape[0])
        for i in range(word.shape[0]):
            print("word[i]", word[i])
            print("kkk", word[i].flatten())
            points = np.append(points, np.array([np.float32(word[i].flatten())]), axis = 0)
    model = KMeans(n_cluster=clusterSize, init='k-means++', max_iter=400, random_state=0)
    return model

def isForceNotZero(flow, threshold = 0.1):
    d = flow[np.where(flow>threshold)]
    return len(d) > 0

def model_lda(kmeans, size):
    videos = os.listdir(NORMAL_DATASET)
    model = lda.LDA(n_topics=size, n_iter=500, random_state=1)
    likelihood=np.array([])
    n = 3
    T = 10

    for video in videos:
      print("Model LDA",video)
      force, _ = socialForce.forceCal(NORMAL_DATASET + "/" + video)
      frCount = force.shape[0]
      dim = ((force.shape[1]-n)/n+1)*((force.shape[2]-n)/n+1)
      arr = np.empty((0, dim))
      for i in xrange(frCount / T):
          f = force[ T*i: T*i + T, :, :, :] #one clip, ten frame
          arr1 = np.array([])
          for k in xrange(0, force.shape[1] -n, n):
              for d in xrange(0, force.shape[2] -n, n):
                  word = f[:, k:k +n, d:d + n,:]
                  cluster = kmeans.predict(np.array([word.flatten()]))
                  arr1 = np.append(arr1, cluster[0])
          arr = np.append(arr, np.array([arr1.copy()]), axis = 0)
          ##arr nesto napraviti sa njim
          for k in range(arr.shape[0]):
              model.fit(arr[k].reshape(11, 15).astype(np.int32))
              likelihood=np.append(likelihood, model.loglikelihood())
    print(likelihood) 

def main():
    n = 5
    T = 10
    C = 10
    K = 30
    videos = os.listdir(NORMAL_DATASET)
    visualW =[]# np.empty((0,n*n*T*3))
    for video in videos:
        print("Video za normalni " + video)
        videoPath = NORMAL_DATASET + '/' + video
        cam = cv2.VideoCapture(videoPath)
        ret, frame = cam.read()
        
        Words = np.zeros((0, T, n, n, 3))
        force, flow = socialForce.forceCal(videoPath)
        frCount = force.shape[0] #brojFramea
        w = np.array([x for x in xrange(0, force.shape[1] - n, n)]) #svaki n-ti
        h = np.array([x for x in xrange(0, force.shape[2] - n, n)]) #svaki n-ti
        rng = frCount / (T-1)
        if frCount % (T-1) == 0:
            rng-=1
        for i in xrange(rng):
            F1 = force[T * i-i:T * i + T-i, :, :, :]  # getting one clip of 10 frames with 1 frame overlap
            Flow1 = flow[T * i-i:T * i + T-i, :, :]
            np.random.shuffle(w)  # random shuffle of array for visual words random selection
            np.random.shuffle(h)
            j = 0
            br = False
            for k in xrange(w.shape[0]):
                for d in xrange(h.shape[0]):
                    Word = F1[:, w[k]:w[k] + n, h[d]:h[d] + n, :]
                    if isForceNotZero(Flow1[:, w[k]:w[k] + n, h[d]:h[d] + n]):
                        Words = np.append(Words, np.array([Word]), axis=0)
                        j += 1
                    if j >= K:
                        br = True
                        break
                if br:
                    break
        #Words
        for i in range(Word.shape[0]):
            visualW.append(np.float32(Word[i].flatten()))
            #=np.append(visualW, np.array([np.float32(Word[i].flatten())]), axis = 0)
    print(visualW)
    modelKM = KMeans(n_clusters=C,init='k-means++',max_iter=500, random_state=0).fit(visualW)
    model_lda(modelKM, K)
    


if __name__ == "__main__":
    main()