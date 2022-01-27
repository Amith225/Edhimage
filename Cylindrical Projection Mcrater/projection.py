from typing import *

import cv2
import numpy as np

# todo: optimize


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

    def show(self):
        cv2.imshow('img', self.image)
        cv2.waitKey(0)


class ImageCluster:
    def __init__(self, imagesArray: List[List[Image]], perWidthScale, perHeightScale):
        self.imArray = np.array(imagesArray)
        self.ws, self.hs = self.imArray.shape[1] * perWidthScale, self.imArray.shape[0] * perHeightScale

        self.avgShape = np.array([0, 0])
        x = y = None
        for y, im in enumerate(self.imArray):
            shapesy = np.array([0, 0])
            for x, i in enumerate(im):
                assert [i.ws, i.hs] == [perWidthScale, perHeightScale], "Distance scales mis-match"
                shapesy += i.image.shape[:2]
            self.avgShape += shapesy // (x + 1)
        self.avgShape = self.avgShape // (y + 1)

    def show(self):
        npArray = []
        for y, im in enumerate(self.imArray[::-1]):
            yArray = []
            for x, i in enumerate(im[::-1]):
                yArray.append(i.image)
            npArray.append(np.concatenate(yArray, axis=1))
        npArray = np.concatenate(npArray)

        shape = self.avgShape
        if (m := max(shape)) > 800:
            shape = (shape * 800 / m).astype(int)
        cv2.imshow('imc', cv2.resize(npArray, shape[::-1]))
        cv2.waitKey(0)


class Projector:
    def __init__(self, radius):
        self.radius = radius

    # todo: implement offset
    def projectToCylinder(self, imageCluster):
        theta = np.array(imageCluster.hs / self.radius)
        fullImage = []
        for y, im in enumerate(imageCluster.imArray):
            newImageX = []
            for x, i in enumerate(im):
                newImageX.append(i.image)
            fullImage.append(np.concatenate(newImageX, axis=1))

        fullImage = np.concatenate(fullImage)
        r = self.projectImg(fullImage, 0, theta).astype(np.uint8)
        cv2.imshow('', cv2.resize(r[::-1], (800, int(800 / r.shape[1] * r.shape[0]))))
        cv2.waitKey(0)

    @staticmethod
    def projectImg(img, thetaStart, thetaEnd):
        thetaStep = (thetaEnd - thetaStart) / img.shape[0]
        thetaPixel = np.arange(start=thetaStart, stop=thetaEnd, step=thetaStep)

        projectionStretch = 1 / np.cos(thetaPixel)
        pixelWidth = int(max(img.shape[1] * projectionStretch))
        newImg = []
        carryRow = np.zeros(img.shape[1:]), 0
        for row, stretch in zip(img[::-1], projectionStretch):
            fillRow = 1 - carryRow[1]
            carryStretch = stretch - fillRow
            newRows = [carryRow[0] + row * fillRow]
            if int(carryStretch):
                newRows.append(np.tile(row, (int(carryStretch), 1)))
            carryStretch -= int(carryStretch)
            carryRow = (row * carryStretch), carryStretch
            nRowShape = np.shape(newRows)
            newRows = cv2.resize(np.array(newRows), (int(nRowShape[1] * stretch), nRowShape[0]))
            extraLen = pixelWidth - newRows.shape[1]
            newRows = np.pad(newRows, ((0, 0), (done := extraLen // 2, extraLen - done), (0, 0)),
                             'constant', constant_values=0)
            newImg.extend(newRows)

        return np.array(newImg)
