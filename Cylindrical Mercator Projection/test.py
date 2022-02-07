import cv2

import projection


def fun():
    im1 = projection.Image.fromFile('sample_large.jpg', 4000, 3000)
    imc = projection.ImageCluster([[im1]*1]*1, im1.ws, im1.hs, im1.image.shape)
    prj = projection.Projector(radius=6400)
    Im2 = prj.projectToCylinder(imc, 27)


# cp.run("fun()")
fun()
