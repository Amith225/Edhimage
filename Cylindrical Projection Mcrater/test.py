import projection
import cProfile as cp


def fun():
    im1 = projection.Image.fromFile('sample_large.jpg', 4000, 3000)
    imc = projection.ImageCluster([[im1]*1]*1, im1.ws, im1.hs, im1.image.shape)
    prj = projection.Projector(6400)
    Im2 = prj.projectToCylinder(imc, -40)
    Im2.show(.5)


# cp.run("fun()")
fun()
