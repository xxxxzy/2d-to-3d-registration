By blender_output-coordinates, it can output the coordinates(3d) of landmarks on 3d mesh and the coordinates(2d) on rendering image.
# TPS:2d image to 3d mesh
Thin Plate Spline(TPS) is a good method to deform 2d to 3d. There is a trick, we change 2d coordinates (x,y) to (x,y,0).

**coefficient(X,Y,lam)**: output the tps function's coefficient.

**defrom(X_test,X_train,Y_train)**: output the 3d coordinates, which is 2d coordinates after tps deformed.

The blue one is tps grid. The black one is the original 3d mesh.

![image](https://github.com/xxxxzy/2d-to-3d-registration/blob/main/2d%20and%203d.png)

# Mesh parameterization
From Wiki:

Given two surfaces with the same topology, a bijective mapping between them exists. On triangular mesh surfaces, the problem of computing this mapping is called mesh parameterization. The parameter domain is the surface that the mesh is mapped onto.

**Flatten(v,f)**: Using Least Squares Conformal Maps from **library libigl** to flatten 3d to 2d uv.

**Flatten_inverse(uv_all, uv_landmark, vn)**: Given 2d uv to get 3d mesh coordinates. Assume it is a linear function on each triangle mesh. And a general algorithm is given from: https://computergraphics.stackexchange.com/questions/8470/how-to-get-the-3d-position-for-the-point-with-0-0-uv-coordinates

Flatten UV:

![image](https://github.com/xxxxzy/2d-to-3d-registration/blob/main/flatten.jpeg)

UV on 3d mesh:

![image](https://github.com/xxxxzy/2d-to-3d-registration/blob/main/3d.jpeg)
