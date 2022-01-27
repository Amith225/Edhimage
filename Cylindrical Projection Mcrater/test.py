import projection
import cProfile as cp


def fun():
    im1 = projection.Image.fromFile('sample_large.jpg', 4000, 3214)
    imc = projection.ImageCluster([[im1, im1], [im1, im1]], 4000, 3214)
    prj = projection.Projector(6400)
    newImc = prj.projectToCylinder(imc)


# cp.run("fun()")
fun()
