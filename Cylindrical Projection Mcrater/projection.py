from typing import *

import cv2
import numpy as np


class Image:
    def __init__(self, array, widthScale, heightScale):
        self.image = array
        self.ws = widthScale
        self.hs = heightScale
        self.px = self.ws / self.image.shape[0]
        self.py = self.hs / self.image.shape[1]

    @staticmethod
    def fromFile(fileName, *args, **kwargs):
        return Image(cv2.imread(fileName), *args, **kwargs)

    def show(self, wait=True):
        cv2.imshow('img', self.image)
        cv2.waitKey(0)
        if wait:
            cv2.waitKey(0)


class ImageCluster:
    def __init__(self, imagesArray: List[List[Image]]):
        self.imArray = np.array(imagesArray)

        self.avgShape = [0, 0]
        self.ws = 0
        self.hs = 0
        for y, im in enumerate(self.imArray):
            shapesy = [0, 0]
            for x, i in enumerate(im):
                shapesy[0] += i.image.shape[0]
                shapesy[1] += i.image.shape[1]
                self.ws += i.ws
            self.hs += i.hs
            self.avgShape[0] += shapesy[0] / (x + 1)
            self.avgShape[1] += shapesy[1] / (x + 1)
        self.avgShape[0] = int(self.avgShape[0] / (y + 1))
        self.avgShape[1] = int(self.avgShape[1] / (y + 1))

    def show(self, wait=True):
        cv2.imshow('imc', cv2.resize(
            np.concatenate(np.concatenate(np.array([[i.image for i in im]
                                                    for im in self.imArray]), axis=1), axis=1), self.avgShape[::-1]))
        if wait:
            cv2.waitKey(0)


class Projector:
    def __init__(self, radius):
        self.radius = radius

    # todo: implement offset
    def projectToCylinder(self, imageCluster):
        theta = np.array(imageCluster.hs / self.radius)
        thetaChunk = theta / imageCluster.imArray.shape[0]
        newImageCluster = []
        for y, im in enumerate(imageCluster.imArray):
            imageClusterY = []
            for x, i in enumerate(im):
                imageClusterY.append(Image(self.projectImg(i, thetaStart := thetaChunk * y, thetaStart + thetaChunk),
                                           i.ws, i.hs))
            newImageCluster.append(imageClusterY)

        return ImageCluster(newImageCluster)

    def projectImg(self, img, thetaStart, thetaEnd):
        thetaStep = (thetaEnd - thetaStart) / img.image.shape[0]
        thetaPixel = np.arange(start=thetaStart, stop=thetaEnd, step=thetaStep)

        projectionStretch = np.round(1 / np.cos(thetaPixel)).astype(int)
        newImg = [np.tile(img.image[0], (projectionStretch[0], 1))]
        for row, stretch in zip(img.image[1:], projectionStretch[1:]):
            newImg = np.concatenate([newImg, np.tile(row, (stretch, 1, 1))])

        return newImg
