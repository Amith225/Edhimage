import projection

im1 = projection.Image.fromFile('sample_large.jpg', 4000, 3214)
# im1.show()
imc = projection.ImageCluster([[im1, im1], [im1, im1]])
# imc.show()
prj = projection.Projector(6400)
newImc = prj.projectToCylinder(imc)
imc.show()
newImc.show()
