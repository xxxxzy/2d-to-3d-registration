#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 00:14:44 2021

@author: xuzhaoyang
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy as sp
import scipy.interpolate
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
import scipy
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from sklearn.neighbors import NearestNeighbors
import igl  
from scipy.interpolate import interp1d
from meshplot import plot, subplot, interact




# X:2d coordinates
# Y:3d coordinates
# n:the number of data you want to use in train set
# i:seed, >0
def data(X,Y,n,i):
    
    p, d = X.shape #p = 60, d = 2
    z = np.zeros(p)
    X = np.c_[X,z.T]
    P = (p-n)/p
    
    X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=P, random_state=i, shuffle=True)
    
    return X_train, X_test, Y_train, Y_test
    


'''
def eta(X,Y,p):
    
    K = np.empty([p,p])
    for i in range (0,p):
        for j in range (0,p):
            r = np.linalg.norm(X[i,:]-Y[j,:])
            K[i,j] = np.linalg.norm(r) ** 3
                   
    return K
'''



def eta(X,Y):
    
    K = np.abs(cdist(X,Y))**3
    
    return K



def coefficient(X,Y,lam):

    p, d = X.shape

    K = eta(X, X)
    P = np.hstack([np.ones((p, 1)), X])


    
    K = K + lam * np.identity(p)
    
    M2 = np.vstack([
        np.hstack([K, P]),
        np.hstack([P.T, np.zeros((d + 1, d + 1))])
    ])
    
    y = np.vstack([Y, np.zeros((d + 1, d))])
    
    #coe = np.linalg.solve(M, y)
    #coe, _, _, _ = np.linalg.lstsq(M2, y, None)
    coe = np.linalg.pinv(M2)@y
    
    return coe



def deform(X_test,X_train,Y_train):
    
    p , d = X_test.shape
    coe = coefficient(X_train,Y_train,0)
    K = eta(X_test, X_train)
    #K = eta(X_train, X_train)
    M = np.hstack([K, np.ones((p, 1)), X_test])
    Y = M@coe
    return Y
 


# X:2d coordinates
# Y:3d coordinates
# n: the size of train data
# t: the time of sampling
def distance(X,Y,n,t):

    # X(x1,x2) to X(x1,x2,0)
    a, d = X.shape #p = 60, d = 2
    z = np.zeros(a)
    X = np.c_[X,z.T]
    
    p = n/30
    
    epsilon = np.empty(t)
    dist = np.empty(n)
    for i in range(1,t+1):
        
        X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=p, random_state=i, shuffle=True)
        x_train,x_test, y_train, y_test = train_test_split(X_test,Y_test,test_size=0.5, random_state=10000, shuffle=True)

        dist = np.diagonal(cdist(deform(x_test,x_train,y_train),y_test))
        dist_max = dist.max()
        dist_mean = np.mean(dist)
        epsilon[i-1] = dist_mean/dist_max
        
    
    return epsilon
    



def surface_curvature(X,Y,Z):

	a, d=X.shape

#First Derivatives
	Xv,Xu=np.gradient(X)
	Yv,Yu=np.gradient(Y)
	Zv,Zu=np.gradient(Z)
#	print(Xu)

#Second Derivatives
	Xuv,Xuu=np.gradient(Xu)
	Yuv,Yuu=np.gradient(Yu)
	Zuv,Zuu=np.gradient(Zu)   

	Xvv,Xuv=np.gradient(Xv)
	Yvv,Yuv=np.gradient(Yv)
	Zvv,Zuv=np.gradient(Zv) 

#2D to 1D conversion 
#Reshape to 1D vectors
	Xu=np.reshape(Xu,a*d)
	Yu=np.reshape(Yu,a*d)
	Zu=np.reshape(Zu,a*d)
	Xv=np.reshape(Xv,a*d)
	Yv=np.reshape(Yv,a*d)
	Zv=np.reshape(Zv,a*d)
	Xuu=np.reshape(Xuu,a*d)
	Yuu=np.reshape(Yuu,a*d)
	Zuu=np.reshape(Zuu,a*d)
	Xuv=np.reshape(Xuv,a*d)
	Yuv=np.reshape(Yuv,a*d)
	Zuv=np.reshape(Zuv,a*d)
	Xvv=np.reshape(Xvv,a*d)
	Yvv=np.reshape(Yvv,a*d)
	Zvv=np.reshape(Zvv,a*d)

	Xu=np.c_[Xu, Yu, Zu]
	Xv=np.c_[Xv, Yv, Zv]
	Xuu=np.c_[Xuu, Yuu, Zuu]
	Xuv=np.c_[Xuv, Yuv, Zuv]
	Xvv=np.c_[Xvv, Yvv, Zvv]
    

#% First fundamental Coeffecients of the surface (E,F,G)
	
	E=np.einsum('ij,ij->i', Xu, Xu) 
	F=np.einsum('ij,ij->i', Xu, Xv) 
	G=np.einsum('ij,ij->i', Xv, Xv) 

	m=np.cross(Xu,Xv,axisa=1, axisb=1) 
	p=np.sqrt(np.einsum('ij,ij->i', m, m)) 
	n=m/np.c_[p,p,p]
    
# n is the normal
#% Second fundamental Coeffecients of the surface (L,M,N), (e,f,g)
	L= np.einsum('ij,ij->i', Xuu, n) #e
	M= np.einsum('ij,ij->i', Xuv, n) #f
	N= np.einsum('ij,ij->i', Xvv, n) #g

# Alternative formula for gaussian curvature in wiki 
# K = det(second fundamental) / det(first fundamental)
#% Gaussian Curvature
	K=(L*N-M**2)/(E*G-F**2)
	K=np.reshape(K,a*d)
	#print(K.size)
#wiki trace of (second fundamental)(first fundamental inverse)
#% Mean Curvature
	H = ((E*N + G*L - 2*F*M)/((E*G - F**2)))/2
	#print(H.shape)
	H = np.reshape(H,a*d)

#% Principle Curvatures
	Pmax = H + np.sqrt(H**2 - K)
	Pmin = H - np.sqrt(H**2 - K)
#[Pmax, Pmin]
	Principle = [Pmax,Pmin]
    
	return H




def surface_region_curvature(xtest,xtrain,ytrain):
    
# use convex hull to find the good region on 2d image
    hull = ConvexHull(xtest[:,0:2])
    ind = hull.vertices
        
# move on the tps surface 
# first find the correspond region points
# second plot the tps surface gird
# thrid find the points in the region
# finally compute the mean curvature

    de = deform(xtest,xtrain,ytrain)
    region_point = de[ind]
    #ch = np.vstack([de[a],de[a[0]]])
    #x, y, z = zip(*ch)


    x_grid = np.linspace(de[:,0].min(), de[:,0].max(), len(de[:,0]))
    y_grid = np.linspace(de[:,1].min(), de[:,1].max(), len(de[:,1]))
    
    spline = sp.interpolate.Rbf(de[:,0],de[:,1],de[:,2],function='cubic',smooth=0)
    
    B1, B2 = np.meshgrid(x_grid, y_grid)
        
    X = B1.reshape(-1)
    Y = B2.reshape(-1)
    
    Z = spline(B1,B2)
    Z1 = Z.reshape(-1)

    lab = np.vstack([X, Y, Z1])
       
    region_tps = Path(region_point[:,0:2]) # make a polygon
    grid = region_tps.contains_points(lab[0:2,:].T)
    np.sum(grid!=0)
    i = [i for i in range(len(grid)) if grid[i] == True]
    
    cur = surface_curvature(B1,B2,Z)
    M = np.sum(np.abs(cur[i]))/len(i)
    
    return M


# X:2d coordinates
# Y:3d coordinates
# n:the number of data you want to use in train set
# i:seed, >0
# m:the time you want to random the the train set    
def mean_surface_curvature(X,Y,n,m):
    
    all_curvature = 0
    for i in range (1,m+1):
        X_train, X_test, Y_train, Y_test = data(X,Y,n,i)
        curvature = surface_region_curvature(X_test,X_train,Y_train)
        all_curvature = all_curvature + curvature
        
    M = all_curvature/m
    
    return M
    
      
    
    
# Output:
#    v[indd]:region points on 3d traingle, the coordinates of 3d traingle
#    de:2d tps landmarks coordinates
#    region_points:points coordinates of convex hull region on tps surface
#    X_test[:,0:2]:2d points
#    Y_test:3d points
#    simplices:for plot 2d image with convell hull
def nearstraingle(X,Y,n,i,v):
    # use convex hull to find the good region on 2d image
    
    X_train, X_test, Y_train, Y_test = data(X,Y,n,i)
    
    hull = ConvexHull(X_test[:,0:2])
    ind = hull.vertices
    simplices = hull.simplices
        

    de = deform(X_test,X_train,Y_train)
    region_point = de[ind]
    
    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(v)
    indd = knn.kneighbors(region_point, 3, return_distance=False)
    
    return v[indd],de,region_point,X_test[:,0:2],Y_test,simplices





# Data processing
# Question can see:
#    https://github.com/libigl/libigl-python-bindings/issues/54
def Processing_3dmesh(v,f):
    
    #components = igl.face_components(f)
    sv,_,_,sf = igl.remove_duplicate_vertices(v,f,1e-7)
    sc = igl.face_components(sf)
    unique, counts = np.unique(sc, return_counts=True)
    dict(zip(unique, counts))
    f_lagest_component = sf[np.where(sc==0)]
    vn,fn,_,_ = igl.remove_unreferenced(sv,f_lagest_component)
    return vn,fn


# LSCM Flatten
def Flatten(v,f):
    
    b = np.array([2, 1])
    bnd = igl.boundary_loop(f)
    b[0] = bnd[0]
    b[1] = bnd[int(bnd.size / 2)]
    bc = np.array([[0.0, 0.0], [1.0, 0.0]])
    ret,uv = igl.lscm(v, f, b, bc)
    return uv


# uv_all: all the uv coordinates
# uv-landmarks: the uv points that you want to compute the 3d coordinates
# vn: all the 3d coordinates
# the code was written by the algorithm from:
#    https://computergraphics.stackexchange.com/questions/8470/how-to-get-the-3d-position-for-the-point-with-0-0-uv-coordinates
def Flatten_inverse(uv_all, uv_landmark, vn):
    
    p,d = uv_landmark.shape
    
    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(uv_all)
    ind = knn.kneighbors(uv_landmark, 3, return_distance=False)
    
    
    uv_traingle = uv_all[ind]
    vn_traingle = vn[ind]
    
    
    
    m1 = uv_traingle[:,1] - uv_traingle[:,0]
    m2 = uv_traingle[:,2] - uv_traingle[:,0]
    
    a = np.vstack([m1,m2]).T
    a[:,[1,2]] = a[:,[2,1]]

    M = a.reshape(p,2,2)
    inv = np.linalg.inv(M)
    
    
    lam = inv@((uv_landmark - uv_traingle[:,0]).reshape(p,2,1))
    
    vertex = (1-lam[:,0]-lam[:,1])*vn_traingle[:,0]+lam[:,0]*vn_traingle[:,1]+lam[:,1]*vn_traingle[:,2]

    return vertex



def spline_image_to_uv(v,uv,point2d,point3d,image_points):
    
    point3d = np.round(point3d,6)
    a = point3d[:,0]
    b = v[:,0]
    
    ind = np.in1d(b,a).nonzero()[0]
    ind2 = np.in1d(a,v[ind,0]).nonzero()[0]

    y = uv[ind]
    x = point2d[ind2]

    xu = x[:,0]
    xv = x[:,1]
    yu = y[:,0]
    yv = y[:,1]

    xu = xu.reshape(xu.shape[0],1)
    yu = yu.reshape(yu.shape[0],1)

    xv = xv.reshape(xv.shape[0],1)
    yv = yv.reshape(yv.shape[0],1)
    
    u = np.hstack([xu,yu])
    u = u[np.lexsort([u.T[0]])]
    
    v = np.hstack([xv,yv])
    v = v[np.lexsort([v.T[0]])]

    cubic_u = interp1d(u[:,0], u[:,1], kind='cubic')
    cubic_v = interp1d(v[:,0], v[:,1], kind='cubic')
    
    u = cubic_u(image_points[:,0])
    v = cubic_v(image_points[:,1])
    
    u = u.reshape(u.shape[0],1)
    v = v.reshape(v.shape[0],1)
    
    uv = np.hstack([u,v])
    
    return uv



def spline_uv_to_image(uv,point2d,uv_points):
    
    
    y = uv
    x = point2d
    
    xu = x[:,0]
    xv = x[:,1]
    yu = y[:,0]
    yv = y[:,1]

    xu = xu.reshape(xu.shape[0],1)
    yu = yu.reshape(yu.shape[0],1)

    xv = xv.reshape(xv.shape[0],1)
    yv = yv.reshape(yv.shape[0],1)
    
    u = np.hstack([xu,yu])
    u = u[np.lexsort([u.T[1]])]
    
    v = np.hstack([xv,yv])
    v = v[np.lexsort([v.T[1]])]

    cubic_u = interp1d(u[:,1], u[:,0], kind='cubic')
    cubic_v = interp1d(v[:,1], v[:,0], kind='cubic')

    x = cubic_u(uv_points[:,0])
    y = cubic_v(uv_points[:,1])
    
    x = x.reshape(x.shape[0],1)
    y = y.reshape(y.shape[0],1)
    
    xy = np.hstack([x,y])

    return xy
    