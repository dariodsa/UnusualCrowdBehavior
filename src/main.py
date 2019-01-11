#!/usr/bin/python

import matplotlib.pyplot as plt
import time
import sys
import os
import cv2
import numpy as np
#import lda
from sklearn.decomposition import LatentDirichletAllocation as LDA
import social_force as socialForce
from sklearn.cluster import KMeans
import visualWords

VIDEO_URL = "../datas/crowd3.avi"
NORMAL_DATASET = "../datas/normalScene2"

def kmeans(clusterSize, visualWords, n=5 , T=10):
    points = np.empty((0,n*n*T*3))
    for word in visualWords:
        for i in range(word.shape[0]):
            points = np.append(points, np.array([np.float32(word[i].flatten())]), axis = 0)
    model = KMeans(n_cluster=clusterSize, init='k-means++', max_iter=400, random_state=0)
    return model

def isForceNotZero(flow, threshold = 0.1):
    d = flow[np.where(flow>threshold)]
    return len(d) > 0

def model_lda(kmeans, size):
    videos = os.listdir(NORMAL_DATASET)
    #model = lda.LDA(n_topics=size, n_iter=100, random_state=1)
    model = LDA(n_components=size)
    likelihood=np.array([])
    n = 5
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
                  #print(len(word.flatten()))
                  cluster = kmeans.predict(np.array([word.flatten()]))
                  print("Cluster",cluster[0])
                  arr1 = np.append(arr1, cluster[0])
          arr = np.append(arr, np.array([arr1.copy()]), axis = 0)
          #arr nesto napraviti sa njim
          print("Arr",arr.shape,arr)
          for k in range(arr.shape[0]):
              model.fit(arr[k].reshape(11, 15).astype(np.int32))
              #likelihood=np.append(likelihood, model.loglikelihood())
    #print(likelihood) 
    return model

def show_data(videoUrl,model,resize=1):
    xdata = []
    ydata = []

    plt.show()

    axes = plt.gca()
    axes.set_xlim(0, 6500)
    axes.set_ylim(-4000, +0) 
    line, = axes.plot(xdata, ydata, 'r-')
    cam = cv2.VideoCapture(videoUrl)
    j=0 
    i=0 
    count=0
    VW = visualWords.getVisualWords(videoUrl, model)
    np.save('words.npy', VW)
def show_data2(videoUrl, resize = 1):
    xdata = []
    ydata = []

    plt.show()
    VW = np.load('words.npy')
    axes = plt.gca()
    axes.set_xlim(0, 6500)
    axes.set_ylim(-4000, +0) 
    line, = axes.plot(xdata, ydata, 'r-')
    cam = cv2.VideoCapture(videoUrl)
    j=0 
    i=0 
    count=0
    print(VW)
    while cam.isOpened():
        ret, img = cam.read()
        if not ret:
            break

        xdata.append(count)
        ydata.append(VW[count])
        line.set_xdata(xdata)
        line.set_ydata(ydata)
        plt.draw()
        plt.pause(1e-17)
        time.sleep(0)
        count+=1

        if j==10:
            i+=1
            j=0
        img = cv2.resize(img, (0, 0), fx=resize, fy=resize)

        cv2.imshow('video', img)
        ch = 0xFF & cv2.waitKey(5)
        print ch
        if ch == 27:
            break
        j+=1
    plt.show()
    cam.release()


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
        print(force.shape, flow.shape)
        for i in xrange(rng):
            F1 = force[  T * i - i:T * i + T-i, :, :, :]  # getting one clip of 10 frames with 1 frame overlap
            Flow1 = flow[T * i - i:T * i + T-i, :, :]
            np.random.shuffle(w)  # random shuffle
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
        for i in range(Words.shape[0]):
            #print("DODAJEM",len(Words[i].flatten()))
            visualW.append(np.float32(Words[i].flatten()))
            #=np.append(visualW, np.array([np.float32(Word[i].flatten())]), axis = 0)
    #print(visualW)
    modelKM = KMeans(n_clusters=C,init='k-means++',max_iter=500, random_state=0).fit(visualW)
    model = model_lda(modelKM, K)
    print("DONE::::::::::::::::::::::::::")
    np.save('model.npy', model)
    show_data(VIDEO_URL, model)


if __name__ == "__main__":
    #model = np.load('model.npy')
    #show_data2(VIDEO_URL)
    main()
