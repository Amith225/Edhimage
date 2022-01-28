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

    def show(self, factor=1):
        cv2.imshow('img', cv2.resize(self.image, (np.array(self.image.shape[:2])[::-1] * factor).astype(int)))
        cv2.waitKey(0)


# todo: take variable sized images
class ImageCluster:
    def __init__(self, imagesArray: List[List[Image]], perWidthScale, perHeightScale, imageShape):
        self.imArray = np.array(imagesArray)
        self.ws, self.hs = self.imArray.shape[1] * perWidthScale, self.imArray.shape[0] * perHeightScale

        self.avgShape = np.array([0, 0])
        x = y = None
        for y, im in enumerate(self.imArray):
            shapesy = np.array([0, 0])
            for x, i in enumerate(im):
                assert [i.ws, i.hs] == [perWidthScale, perHeightScale], "Distance scales mis-match"
                assert i.image.shape == imageShape, "Image shape mis-match"
                shapesy += i.image.shape[:2]
            self.avgShape += shapesy // (x + 1)
        self.avgShape = self.avgShape // (y + 1)

    def show(self, factor=1):
        npArray = []
        for y, im in enumerate(self.imArray[::-1]):
            yArray = []
            for x, i in enumerate(im[::-1]):
                yArray.append(i.image)
            npArray.append(np.concatenate(yArray, axis=1))
        npArray = np.concatenate(npArray)

        cv2.imshow('imc', cv2.resize(npArray, (factor * self.avgShape[::-1]).astype(int)))
        cv2.waitKey(0)


class Projector:
    def __init__(self, radius):
        self.radius = radius

    def projectToCylinder(self, imageCluster, yOffsetAngle=0):
        yOffsetAngle = np.radians(yOffsetAngle)
        theta = np.array(imageCluster.hs / self.radius)
        fullImage = []
        for y, im in enumerate(imageCluster.imArray):
            newImageX = []
            for x, i in enumerate(im):
                newImageX.append(i.image)
            fullImage.append(np.concatenate(newImageX, axis=1))

        fullImage = np.concatenate(fullImage)

        return Image(self.projectImg(fullImage, yOffsetAngle, theta + yOffsetAngle)[::-1].astype(np.uint8), 0, 0)

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
                newRows.extend(np.tile(row, (int(carryStretch), 1, 1)))
            newRows = np.array(newRows)
            carryStretch -= int(carryStretch)
            carryRow = (row * carryStretch), carryStretch
            newRows = cv2.resize(newRows, (int(newRows.shape[1] * stretch), newRows.shape[0]))
            extraLen = pixelWidth - newRows.shape[1]
            newRows = np.pad(newRows, ((0, 0), (done := extraLen // 2, extraLen - done), (0, 0)),
                             'constant', constant_values=0)
            newImg.extend(newRows)

        return np.array(newImg)
