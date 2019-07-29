# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
from matplotlib import path
import matplotlib.pylab as plt
from pykrige.ok import OrdinaryKriging
import time

# Define function for extracting points
def _polygon_wam(polygon_vertices, N):
    tri, err = _minimal_triangulation(polygon_vertices[:,0], polygon_vertices[:,1])
    tri = np.matrix(tri); 
    ref_pts = _ref_wam_pts(N)
    wam_ptsx, wam_ptsy = [], []
    for index_triangulation in range(tri.shape[0]):
        loc_tri_indices = tri[index_triangulation,:]
        loc_polygon_vertices = polygon_vertices[loc_tri_indices.astype(int)-1, :]
        loc_triangle_pts = _triangle_map(loc_polygon_vertices, ref_pts)
        wam_ptsx.append(np.array(loc_triangle_pts[:,0].T))
        wam_ptsy.append(np.array(loc_triangle_pts[:,1].T))
    
    wam_pts = np.vstack((np.hstack(wam_ptsx), np.hstack(wam_ptsy)))
    return np.matrix(wam_pts.T), err
    
    
# Define auxiliary functions for constructing the triangulation
def _triangle_map(triangle_vertices,pts_in):
    Q1, Q2, Q3 = triangle_vertices[:,0], triangle_vertices[:,1], triangle_vertices[:,2]
    P = np.vstack(( (Q2 - Q1), (Q3 - Q1) )).T
    pts_out_pre = (P@pts_in.T).T
    res = np.hstack((Q1.T[0] + pts_out_pre[:,0], Q1.T[1] + pts_out_pre[:,1]))
    return res
    
    
# Refine the mesh        
def _ref_wam_pts(deg):
    n1 = 2*deg
    triangle = np.array([[0, 1],[1, 0],[0, 0]])
    j1, j2 = np.arange(deg), np.arange(deg+1)
    rho, theta = np.meshgrid(np.cos(j1*np.pi / n1), j2*np.pi / n1)
    rho1 = np.array([rho[:,i] for i in range(rho.shape[1])]).flatten()
    theta1 = np.array([theta[:,i] for i in range(theta.shape[1])]).flatten()
    B1, B2 = rho1*np.cos(theta1), rho1*np.sin(theta1)
    meshS = np.matrix(np.vstack((B1**2, B2**2)).T)  
    meshT1 = triangle[2,0] * (1-meshS[:,0] - meshS[:,1]) + triangle[1,0] * meshS[:,1] + triangle[0,0] * meshS[:,0]   
    meshT2 = triangle[2,1] * (1-meshS[:,0] - meshS[:,1]) + triangle[1,1] * meshS[:,1] + triangle[0,1] * meshS[:,0]
    nodes_x = np.vstack((meshT1,[0])); nodes_y = np.vstack((meshT2,[0]))
    return np.hstack((nodes_x, nodes_y))
    
    
# Compute minimal triangulation
def _minimal_triangulation(x, y):
    pts = np.vstack((x, y)).T
    pts_remaining = pts.copy()
    pointer_pts_remaining = np.linspace(1, len(x.T), len(x.T))
    tri, err = [], 0
    while len(pts_remaining[:,0]) > 3:
        tri_loc, err = _find_ear(pts_remaining)
        tri_loc = tri_loc.astype(int)
        tri_add = pointer_pts_remaining[tri_loc-1]
        tri = np.hstack((tri, tri_add))
        pointer_pt_remove = tri_add[1]
        pointer_pts_remaining = np.setdiff1d(pointer_pts_remaining,pointer_pt_remove)
        pointer_pts_remaining = pointer_pts_remaining.astype(int)
        pts_remaining = pts[pointer_pts_remaining-1,:]

    tri = np.hstack((tri,pointer_pts_remaining))
    tri = np.reshape(tri, (int(tri.shape[0]/tri_loc.shape[0]),tri_loc.shape[0]))
    return tri, err
    
    
# Compute the needed triangles
def _find_ear(vertices_pts):
    number_pts = vertices_pts.shape[0]; 
    a = np.linspace(1, number_pts,number_pts)
    vertices_index = np.hstack((a, [1])).T;
    pts_remaining = vertices_pts
    curr_starting_index, ear_found, err = 0, 0, 0
    while (ear_found == 0) and (curr_starting_index < number_pts-1):
        trial_triangle_vertices_pointer = vertices_index[curr_starting_index:curr_starting_index+3]
        vertices_index = vertices_index.astype(int)
        trial_triangle_vertices_pointer = trial_triangle_vertices_pointer.astype(int)
        pts_remaining_pointer = np.setdiff1d(np.linspace(1,number_pts,number_pts),\
                                np.linspace(curr_starting_index+1,curr_starting_index+3, \
                                curr_starting_index+3))
        pts_remaining = vertices_pts[pts_remaining_pointer.astype(int)-1, :]
        xv = np.hstack((vertices_pts[trial_triangle_vertices_pointer-1, 0], vertices_pts[curr_starting_index, 0]))
        yv = np.hstack((vertices_pts[trial_triangle_vertices_pointer-1, 1], vertices_pts[curr_starting_index, 1]))
        pr1 = np.sum(vertices_pts[trial_triangle_vertices_pointer-1, 0]) / 3
        pr2 = np.sum(vertices_pts[trial_triangle_vertices_pointer-1, 1]) / 3
        barycenter_triangle = np.hstack((pr1, pr2))
        xt, yt = pts_remaining[:,0], pts_remaining[:,1]
        polygon = np.vstack((xv, yv))
        bt = np.vstack((xt, yt))
        pat = path.Path(polygon.T)
        polygon = np.vstack((vertices_pts[vertices_index-1,0], vertices_pts[vertices_index-1, 1]))
        bt = np.vstack((barycenter_triangle[0], barycenter_triangle[1]))
        pat = path.Path(polygon.T) 
        in2 = pat.contains_points(bt.T)
        if (np.sum(in2[0:-2]) == 0):
            ear_found, err = 1, 0
            tri = np.linspace(curr_starting_index+1, curr_starting_index+3,curr_starting_index+3)
        else:
            err = 1
            break
        curr_starting_index += 1
    return tri, err
    
    
# Construct the Vandermonde matrix 
def _chebvand(deg, wam, rect):
    a, b, c, d = rect
    j1, j2 = np.meshgrid(range(deg+1),range(deg+1))
    j11 = np.array([j1[:,i] for i in range(j1.shape[1])]).flatten()
    j22 = np.array([j2[:,i] for i in range(j2.shape[1])]).flatten()
    good = np.argwhere(j11+j22 < deg+1)
    couples = np.matrix(np.vstack((j11[good].T, j22[good].T)).T)
    mappa1 = np.array((2 * wam[:,0] - b - a) / (b - a))
    mappa2 = np.array((2 * wam[:,1] - d - c) / (d - c))
    mappa = np.vstack((np.array(mappa1.T), np.array(mappa2.T))).T
    mappa = np.abs(mappa + 1) - 1
    V1 = np.cos(np.multiply(couples[:,0], np.arccos(mappa[:,0].T)))
    V2 = np.cos(np.multiply(couples[:,1], np.arccos(mappa[:,1].T)))
    return np.multiply(V1, V2).T
    
    
# Compute the coefficients for polynomial approximation
def _wamfit(deg,wam,pts,fval):
    both = np.vstack((wam,pts))
    rect0, rect1 = np.min(both[:,0]), np.max(both[:,0])
    rect2, rect3 = np.min(both[:,1]), np.max(both[:,1])
    rect = [rect0, rect1, rect2, rect3]
    Q, R1, R2 = _wamdop(deg, wam, rect)
    cfs = np.matmul(Q.T, fval)
    DOP = _wamdopeval(deg, R1, R2, pts, rect)
    lsp = np.matmul(DOP,cfs)
    return cfs, lsp
    
    
# Evaluate the interpolant
def _wamdop(deg,wam,rect):
    V = _chebvand(deg,wam,rect)
    Q1, R1 = np.linalg.qr(V)
    TT = np.linalg.solve(R1.T,V.T).T
    Q, R2 = np.array(np.linalg.qr(TT))
    return Q, R1, R2
    
    
# Evaluate the interpolant
def _wamdopeval(deg,R1,R2,pts,rect):
    W = _chebvand(deg,pts,rect)
    TT = np.linalg.solve(R1.T, W.T).T
    DOP = np.linalg.solve(R2.T, TT.T).T
    return DOP
    
    
# Define the compression rule, here nnls
def _compression_rule(Q,orthmom,X,omega,deg,pos):
    weights = scipy.optimize.nnls(Q.T,np.array(np.ravel(orthmom)).T)[0]
    np.argwhere(weights[:,] > 0)
    return weights
    
    
# Compute points and weights
def _compresscub_extended(deg,X,omega,pos,rect):
    V=_chebvand(deg, X, rect)
    Q, R = np.linalg.qr(V)
    Q = np.real(Q)
    orthmom = Q.T@omega
    weights = _compression_rule(Q, orthmom, X, omega, deg, pos).T
    ind = np.argwhere(weights > 0) 
    return X[ind], weights[ind], Q, R, ind


def plot_image(img, fig_name, pts=None, pts1=None):
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    if pts:
        plt.plot(pts[0], pts[1], 'r-')
    if pts1 :
        plt.plot(pts1[0], pts1[1], 'r.')
    plt.savefig('%s.eps' %fig_name,bbox_inches='tight')
    plt.savefig('%s.png' %fig_name,bbox_inches='tight')    
    plt.close()       


def plot_error(cr, err, name_cr, name_err, fig_name):
    plt.plot(cr, err, 'o-')
    plt.xlabel(name_cr)
    plt.ylabel(name_err)
    plt.savefig('%s.eps' %fig_name, bbox_inches='tight')
    plt.savefig('%s.png' %fig_name, bbox_inches='tight')
    plt.close()


def plot_variogram(krig, ylim, fig_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(krig.lags, krig.semivariance, 'k*')
    plt.ylim(ylim)
    plt.xlabel('Lag distance')
    plt.ylabel('Semivariance')
    plt.savefig('%s.eps'%fig_name)
    plt.savefig('%s.png'%fig_name)
    plt.close()
    

def psnr(im1, im2, N):
    mse = np.sum((np.power(im1-im2,2))) / N
    if mse == 0: return 100
    pixelmax = 255.0
    return 20 * np.log10(pixelmax / np.sqrt(mse))
    
    
def mse(im1, im2, N):
    return np.sum((np.power(im1-im2, 2))) / N
    

# Define the RBF
def _rbfm(ep, r):
    return np.exp(-ep*r)
    
     
# Load the image and define the polygon
matr = scipy.io.loadmat('sm_simulata.mat')
Image_large = matr["image_temp"]

plot_image(Image_large, 'Simulated_ImageLarge_CT')

lx, rx, ly, ry = 80, 375, 79, 361
Image = Image_large[lx-1:rx, ly-1:ry]
m, n = Image.shape
sx, sy = Image.shape
scaling = np.array([[1, 0],[0, 1]])
vertex = np.array([[100,70,100,130,160,180], [250,220,220,150,150,250]])

polygon = (scaling@vertex).T
xM, yM = np.max(polygon, axis=0) 
xm, ym = np.min(polygon, axis=0)
rect = [xm, xM, ym, yM]

polygonclosed = np.vstack((polygon, polygon[0,:]))
xpolygon, ypolygon = polygonclosed[:,0], polygonclosed[:,1]

# Define the polynomial degree
degs = [10,15,20,25,30,35,40,45]

# Initialize
CR_VSDK, CR_POLY = [], []
MSE_VSDK, MSE_POLY = [], []
PSNR_VSDK, PSNR_POLY = [], []


for idx, deg in enumerate(degs):
    print(idx, deg)
    
    # Define the evaluation points 
    imagepointsX, imagepointsY = np.meshgrid(np.linspace(1, sx, sx), np.linspace(1, sy, sy)); 
    xxa = np.tile(imagepointsX, (1, 1)); 
    x = np.array([imagepointsX[:,i] for i in range(imagepointsX.shape[1])]).flatten()
    y = np.array([imagepointsY[:,i] for i in range(imagepointsY.shape[1])]).flatten()
    
    imagepointsX, imagepointsY = y.copy(), x.copy()
    
    bt = np.array(np.vstack((imagepointsX, imagepointsY)).T)
    p = path.Path(polygonclosed)
    samplepointsIndex = p.contains_points(bt)
    g0 = Image.flatten()
    
    Imagepiece = np.ones((sx, sy))
    samplepointsIndex = np.where(samplepointsIndex==1)
    g1 = Imagepiece.flatten()
    g1[samplepointsIndex] = g0[samplepointsIndex]
    
    Imagepiece = np.reshape(g1, (sx,sy))
    samplepoints = bt[samplepointsIndex]
    samplepointsx = bt[samplepointsIndex][:, 0]
    samplepointsy = bt[samplepointsIndex][:, 1]
    
    t1 = time.time()
    
    # Compute the TC points
    wam_pts, err = _polygon_wam(polygon, 2*deg)
    samplepointsIndexr = p.contains_points(wam_pts)
    wam_pts = wam_pts[samplepointsIndexr, :]
    xpts1, ypts1 = np.floor(wam_pts[:,0]), np.floor(wam_pts[:,1])
    pts1 = np.hstack((xpts1, ypts1)).astype(int)
    
    pts, w, Q, R, ind = _compresscub_extended(2*deg, pts1, np.ones((len(xpts1),1)), 1, rect)
    pts = pts[:, 0, :]
    pts, ind = np.unique(pts, axis=0, return_index=True)
    xpts, ypts = pts[:,0], pts[:,1]
    
    # Compute the values at points
    fval = np.matrix([Image[int(np.floor(ypts[i])),int(np.floor(xpts[i]))] for i in range(xpts.shape[0])]).T
    
    # Compute the polynomial approximant
    cfs, lsp = _wamfit(deg, pts, samplepoints,fval)
    Imageapprox = np.ones((sx, sy)).T
    g1 = Imageapprox.flatten()
    samplepointsIndex1 = np.array(samplepointsIndex)
    g1[samplepointsIndex1] = lsp.T
    Imageapprox = np.reshape(g1, (sx,sy))

    t2 = time.time()
    
    # Comparisons with VSDKs
    wam_pts, err = _polygon_wam(polygon,1*deg)
    samplepointsIndexr = p.contains_points(wam_pts)
    wam_pts = wam_pts[samplepointsIndexr,:]
    xpts1 = np.floor(wam_pts[:,0]); ypts1 = np.floor(wam_pts[:,1]);
    P = np.hstack((xpts1, ypts1))
    pts1 = np.matrix((P))
    pts1 = pts1.astype(int)
    pts,w,Q,R,ind = _compresscub_extended(1*deg,pts1,np.ones((len(xpts1),1)),1,rect);
    pts =pts[:,0,:]; pts, ind = np.unique(pts, axis=0, return_index=True)
    xpts=pts[:,0]; ypts=pts[:,1]; xpts1=np.floor(wam_pts[:,0])
    ypts1 = np.floor(wam_pts[:,1]); P = np.hstack((xpts1,ypts1))
    pts1 = np.matrix((P)); pts1 = pts1.astype(int)
    
    # Compute the values at points
    fval = np.matrix([Image[int(np.floor(ypts[i])),int(np.floor(xpts[i]))] for i in range(xpts.shape[0])]).T
    
    xpts, ypts = pts[:,0], pts[:,1]
    ptsa = pts.copy()
    
    # Define the mask for VSDKs    
    wam1 = np.matrix(np.vstack((np.array((xpts/sy).T), np.array((ypts/sx).T))).T)
    pts_1 = np.matrix(np.vstack((np.array(samplepointsx), np.array(samplepointsy))).T)
    pts = np.matrix(np.vstack((np.array(samplepointsx/sy), np.array(samplepointsy/sx))).T)
    
    # Construct the augmented set of nodes and evaluation points
    dsites = np.hstack((wam1, fval));
    fval3 = [Image[int(np.floor(samplepointsy[i])), int(np.floor(samplepointsx[i]))] for i in range(pts.shape[0])]
    fval3 = np.matrix(fval3).T            
    
    add_ep1 = Image.flatten()[samplepointsIndex];
    super_threshold_indices = add_ep1 > 0
    add_ep = 0*Image.flatten()[samplepointsIndex]; add_ep[super_threshold_indices] = 1
    add_ep2 = fval
    super_threshold_indices = add_ep2 > 0
    add_ep3 = 0*fval; add_ep3[super_threshold_indices] = 1
 
    dsites = np.hstack((wam1,add_ep3));
    epoints = np.hstack((pts,np.matrix(add_ep).T))
    
    # Compute kernel and evaluation matrices
    NN = dsites.shape[0]
    DM = np.zeros((NN, NN))        
    NN1 = epoints.shape[0]
    DM_eval = np.zeros((NN1, NN))     
    for count in range(3):
        dr, cc = np.meshgrid(epoints[:,count], dsites[:,count])
        DM_eval = DM_eval + (np.power((dr-cc),2)).T
        dr, cc = np.meshgrid(dsites[:,count], dsites[:,count]); 
        DM = DM + (np.power((dr-cc), 2)).T
    IM1 = _rbfm(1, np.sqrt(DM))
    EM1 = _rbfm(1, np.sqrt(DM_eval))
    
    # Compute and evaluate the VSDKs interpolant
    coef = np.linalg.solve(IM1, fval)
    Pf = (EM1.dot(coef))
    
    #OK
    t3 = time.time()
    
    # Compute accuracy indicators (psnr and mse)
    Imageapprox1 = Image.flatten();
    Imageapprox1[samplepointsIndex] = Pf.T
    
    Imageapproxs1 = np.reshape(Imageapprox1, (sx,sy))
    
    Imageapprox = Image.flatten()
    Imageapprox[samplepointsIndex] = lsp.T
    Imageapproxs = np.reshape(Imageapprox, (sx,sy))
    #OK
    
    snr1 = psnr(Image, Imageapproxs1, (Pf.shape[0]))
    rmse1 = mse(Imageapproxs1, Image, (Pf.shape[0]))
    snr = psnr(Image, Imageapproxs, (Pf.shape[0]))
    rmse = mse(Imageapproxs, Image, (Pf.shape[0]))

    # Variograms
    incr = 10
    fvalk, fvalkp, fvalkv, ptk1, ptk2 = [], [], [], [], []
    for i in range(0, pts.shape[0], incr):
        ptk1.append(pts_1[i, 1])
        ptk2.append(pts_1[i, 0])
        fvalkp.append(lsp[i, 0])
        fvalkv.append(Pf[i, 0])
        fvalk.append(fval3[i, 0])
    
    OK = OrdinaryKriging(ptk1, ptk2, fvalk, variogram_model='spherical', nlags=30) 
    OK1 = OrdinaryKriging(ptk1, ptk2, fvalkp, variogram_model='spherical', nlags=30)
    OK2 = OrdinaryKriging(ptk1, ptk2, fvalkv, variogram_model='spherical', nlags=30)
    OK3 = OrdinaryKriging(ptsa[:,0], ptsa[:,1], fval, variogram_model='spherical', nlags=30) 
        
    CR_VSDK.append(Pf.shape[0] / coef.shape[0])
    CR_POLY.append(Pf.shape[0] / cfs.shape[0])
    MSE_VSDK.append(rmse1)
    MSE_POLY.append(rmse)
    PSNR_VSDK.append(snr1)
    PSNR_POLY.append(snr)
    
    #TEMP
    print(rmse1)
    print(rmse)
    #

    plot_variogram(OK,  (.000005,0.00045), 'Sim_CT_variogram_orig_im')
    plot_variogram(OK1, (.000005,0.00045), 'Sim_CT_variogram_poly_%d' %deg)
    plot_variogram(OK2, (.000005,0.00045), 'Sim_CT_variogram_VSDK_%d' %deg)
    plot_variogram(OK3, (.000005,0.00045), 'Siml_CT_variogram_points_%d' %deg)

    Outfile = open('Log_CT_SimImage_%d.txt' %deg, "w")
    out_string = 'Partial Sill for original image:%s\n' %(OK.variogram_model_parameters[0])
    out_string += 'Full Sill for original image:%s\n' %(OK.variogram_model_parameters[0]+OK.variogram_model_parameters[2])
    out_string += 'Range for original image:%s\n' %(OK.variogram_model_parameters[1])
    out_string += 'Nugget for original image:%s\n' %(OK.variogram_model_parameters[2])
    out_string += 'Partial Sill for polynomial image:%s\n' %(OK1.variogram_model_parameters[0])
    out_string += 'Full Sill for polynomial image:%s\n' %(OK1.variogram_model_parameters[0]+OK1.variogram_model_parameters[2])
    out_string += 'Range for polynomial image:%s\n' %(OK1.variogram_model_parameters[1])
    out_string += 'Nugget for polynomial image:%s\n' %(OK1.variogram_model_parameters[2])
    out_string += 'Partial Sill for VSDK image:%s\n' %(OK2.variogram_model_parameters[0])
    out_string += 'Full Sill for VSDK image:%s\n' %(OK2.variogram_model_parameters[0]+OK2.variogram_model_parameters[2])
    out_string += 'Range for VSDK image:%s\n' %(OK2.variogram_model_parameters[1])
    out_string += 'Nugget for VSDK image:%s\n' %(OK2.variogram_model_parameters[2])
    out_string += 'Partial Sill for VSDK points:%s\n' %(OK3.variogram_model_parameters[0])
    out_string += 'Full Sill for VSDK points:%s\n' %(OK3.variogram_model_parameters[0]+OK3.variogram_model_parameters[2])
    out_string += 'Range for VSDK points:%s\n' %(OK3.variogram_model_parameters[1])
    out_string += 'Nugget for VSDK points:%s\n' %(OK3.variogram_model_parameters[2])
    out_string += 'Original number of points:%s\n' %(Pf.shape[0])
    out_string += 'Number of CT points for polynomials:%s\n' %(wam_pts.shape[0])
    out_string += 'Number of coefficients for polynomials:%s\n' %(cfs.shape[0])
    out_string += 'Compress ratio for polynomials:%s\n' %( Pf.shape[0]/cfs.shape[0])
    out_string += 'Number of CT points for VSDKs:%s\n' %(wam1.shape[0])
    out_string += 'Number of coefficients for VSDKs:%s\n' %(coef.shape[0])
    out_string += 'Compress ratio for VSDK:%s\n' %(Pf.shape[0]/coef.shape[0])
    out_string += 'CPU time for polynomials:%s\n' %(t2-t1)
    out_string += 'CPU time for VSDKs:%s\n' %(t3-t2)
    out_string += 'PSNR for polynomials:%s\n' %(snr)
    out_string += 'PSNR for VSDKs:%s\n' %(snr1)
    out_string += 'MSE for polynomials:%s\n' %(rmse)
    out_string += 'MSE for VSDKs:%s\n' %(rmse1)
    Outfile.write(out_string)   
    Outfile.close()

    # Save data
    np.savetxt('Sim_polyghon_OriginalImage_xy', bt)
    np.savetxt('Sim_polyghon_OriginalImage_z', Image)
    np.savetxt('Sim_polyghon_ApproximatedImage_TC_z_%d' %deg, Imageapprox)
    np.savetxt('Sim_polyghon_ApproximatedImage_TC_VSDK_z_%d' %deg, Imageapprox1)
    np.savetxt('Sim_polyghon_OriginalImage_TC_xy_%d' %deg, pts1)
    np.savetxt('Sim_polyghon_OriginalImage_TC_xy_VSDK_%d' %deg, wam1)
 
    # Save images
    lx, rx, ly, ry = 130, 275, 49, 200
    
    # Save images
    plot_image(Imageapproxs1[lx-1:rx,ly-1:ry], 'Sim_ApproximatedImage_CT_VSDK_%d' %deg, pts=(xpolygon-ly,ypolygon-lx))
    plot_image(Imageapproxs[lx-1:rx,ly-1:ry], 'Sim_ApproximatedImage_CT_%d' %deg, pts=(xpolygon-ly,ypolygon-lx))
    plot_image(Image[lx-1:rx,ly-1:ry], 'Sim_OriginalImage_CT_zoom', pts=(xpolygon-ly,ypolygon-lx))
    plot_image(Image, 'Sim_OriginalImage_CT_%d' %deg, pts=(xpolygon,ypolygon),pts1=(xpts,ypts))

plot_error(CR_VSDK, MSE_VSDK,'CR','MSE','SimCR_VS_MSE_CT_VSDK')
plot_error(CR_POLY, MSE_POLY,'CR','MSE','SimCR_VS_MSE_CT')
plot_error(CR_VSDK, PSNR_VSDK,'CR','PSNR','SimCR_VS_PSNR_CT_VSDK')
plot_error(CR_POLY, PSNR_POLY,'CR','PSNR','SimCR_VS_PSNR_CT')
