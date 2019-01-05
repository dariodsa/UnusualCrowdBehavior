#!/usr/bin/python

import cv2
import numpy as np

def biinterpolation(x, y, flow):
    im=x if x-1<0 else x-1
    ip=x if x+1>=flow.shape[0] else x+1
    jm=y if y-1<0 else y-1
    jp=y if y+1>=flow.shape[1] else y+1

    point1x=flow[im,jm,0]
    point1y=flow[im,jm,1]

    point2x = flow[im, jp, 0]
    point2y = flow[im, jp, 1]

    point3x = flow[ip, jm, 0]
    point3y = flow[ip, jm, 1]

    point4x = flow[ip, jp, 0]
    point4y = flow[ip, jp, 1]

    Vx=(point1x*(x-im)*(y-jm)+point2x*(x-im)*(jp-y)+point3x*(ip-x)*(y-jm)+point4x*(ip-x)*(jp-y))/((ip-im)*(jp-jm))
    Vy = (point1y * (x - im) * (y - jm) + point2y * (x - im) * (jp - y) + point3y * (ip - x) * (y - jm) + point4y * (
ip - x) * (jp - y)) / ((ip - im) * (jp - jm))
    return Vx, Vy

def effectiveVelocity(flow):
    sp = flow.shape
    O = np.zeros((sp[0], sp[1], sp[2]))
    for r in range(sp[0]):
        for c in range(sp[1]):
            Vx, Vy = biinterpolation(r, c, flow)
            O[r, c, 0] = Vx
            O[r, c, 1] = Vy
    return O

def getForceFromImage(force):
    print("Force",force)
    F=np.sqrt(force[:,:,0]**2+force[:,:,1]**2)
    F=F[1:F.shape[0]-1,1:F.shape[1]-1]  #removing corner pixels  these have not good bilinear interpolation velocities
    F*=255.0/np.max(F)
    return np.round(F).astype(np.uint8)


def forceCal(videoPath, tau = 0.5, Pi = 0):
    cam = cv2.VideoCapture(videoPath)
    ret, prev = cam.read()

    prev = cv2.resize(prev, (0,0), fx = 0.25, fy = 0.25)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevflow = np.zeros((prev.shape[0], prev.shape[1], 2))
    result = np.empty((0, prev.shape[0] - 2, prev.shape[1] - 2,3))
    forceRes = np.empty((0, prev.shape[0] - 2, prev.shape[1] -2))
    fps = cam.get(cv2.CAP_PROP_FPS)  # frames per second
    
    while (cam.isOpened()):
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_lucas_kanade.html
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.1, flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        
        Vef=effectiveVelocity(flow)
        Vq = (1-Pi)*Vef+Pi*flow       #desired velosity Vea
        F = tau*(Vq-flow)-(flow-prevflow)/1/fps
        
        F1=getForceFromImage(F)
        imC = cv2.applyColorMap(F1, cv2.COLORMAP_JET) #from BLUE TO RED, temp
        result=np.append(result,np.array([imC]),axis=0) #dodaj u niz
        forceRes = np.append(forceRes, np.array([F1]), axis=0) #dodaj F1
        prevflow=flow
        prevgray = gray
    cam.release()
    return result,forceRes
