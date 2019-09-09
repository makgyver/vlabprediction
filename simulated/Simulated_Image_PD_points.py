# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import matplotlib.pylab as plt
from pykrige.ok import OrdinaryKriging
import time

# Define the function for computing Padova points
def _pdpts(n):
    zn  = np.cos(np.linspace(0, 1, n+1)*np.pi)
    zn1 = np.cos(np.linspace(0, 1, n+2)*np.pi)
    Pad1, Pad2 = np.meshgrid(zn, zn1)
    f1 = np.linspace(0, n, n+1)
    f2 = np.linspace(0, n+1, n+2)
    M1, M2 = np.meshgrid(f1,f2)
    h = np.array(np.mod(M1 + M2, 2))
    g = np.array(np.concatenate(h.T))
    findM = np.argwhere(g)
    Pad_x = np.matrix(np.concatenate(Pad1.T)[findM])
    Pad_y = np.matrix(np.concatenate(Pad2.T)[findM])
    return Pad_x, Pad_y
    
    
# Compute the coefficients for polynomial approximation
def _wamfit(deg, wam, pts, fval):
    both = np.vstack((wam, pts))
    rect = [np.min(both[:,0]), np.max(both[:,0]), np.min(both[:,1]), np.max(both[:,1])]
    Q, R1, R2 = _wamdop(deg, wam, rect)
    DOP = _wamdopeval(deg, R1, R2, pts, rect)
    cfs = np.matmul(Q.T, fval)
    lsp = np.matmul(DOP, cfs)
    return cfs, lsp
    
    
# Evaluate the approximant
def _wamdopeval(deg,R1,R2,pts,rect):
    W = _chebvand(deg, pts, rect)
    TT = np.linalg.solve(R1.T, W.T).T
    return np.linalg.solve(R2.T, TT.T).T
    
    
# Factorize the Vandermonde matrix
def _wamdop(deg,wam,rect):
    V = _chebvand(deg,wam,rect)
    Q1, R1 = np.linalg.qr(V)
    TT = np.linalg.solve(R1.T,V.T).T
    Q, R2 = np.array(np.linalg.qr(TT))
    return Q, R1, R2
    
    
# Construct the Vandermonde matrix
def _chebvand(deg,wam,rect):
    j = np.linspace(0, deg, deg+1)
    j1, j2 = np.meshgrid(j, j)
    j11 = j1.T.flatten()
    j22 = j2.T.flatten()
        
    good = np.argwhere(j11+j22 < deg+1)
    couples = np.matrix(np.vstack((j11[good].T, j22[good].T)).T)
    a, b, c, d = rect
    mappa1 = (2.* wam[:, 0] - b - a) / (b - a)
    mappa2 = (2.* wam[:, 1] - d - c) / (d - c)
    mappa = np.vstack((mappa1.T, mappa2.T)).T
    V1 = np.cos(np.multiply(couples[:,0], np.arccos(mappa[:,0].T)))
    V2 = np.cos(np.multiply(couples[:,1], np.arccos(mappa[:,1].T)))
    V = np.multiply(V1, V2).T
    return V
    
    
# This function computes a number of Padova points for comparing VSDK and
# polynomial approximation
def _pdpts_vsdk(n, n1):
    Pad_x = np.array([0])
    while Pad_x.shape[0] < n1:
        zn  = np.cos(np.linspace(0, 1, n+1) * np.pi);
        zn1 = np.cos(np.linspace(0, 1, n+2) * np.pi);
        Pad1, Pad2 = np.meshgrid(zn, zn1)
        M1, M2 = np.meshgrid(np.linspace(0, n, n+1), np.linspace(0, n+1, n+2))
        findM = np.argwhere(np.concatenate(np.mod(M1 + M2, 2).T))
        Pad_x = np.concatenate(Pad1.T)[findM]
        Pad_y = np.concatenate(Pad2.T)[findM]
        n += 1
        
    return Pad_x, Pad_y
    
    
# Define the RBF
def _rbfm(ep, r):
    return np.exp(-ep*r)
    
    
# Compute accuracy indicators (psnr and mse)
def psnr(im1, im2):
    mse = np.mean(np.power(im1-im2, 2))
    if mse == 0: return 100
    pixelmax = 255.0
    return 20 * np.log10(pixelmax / np.sqrt(mse))
    
    
def mse(im1, im2):
    return np.mean(np.power(im1-im2, 2))
    

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
  
def plot_image(img, fig_name, pts=None):
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    if pts:
        plt.plot(pts[0], pts[1], 'r.')
    plt.savefig('%s.eps' %fig_name,bbox_inches='tight')
    plt.savefig('%s.png' %fig_name,bbox_inches='tight')
    plt.close()
  
# Load the image
matr = scipy.io.loadmat('sm_simulata.mat')
Image_large = matr["image_temp"]

plot_image(Image_large, 'Simulated_ImageLarge')

sx, sy = Image_large.shape
lx, rx, ly, ry = 170, 275, 79, 361
Image = Image_large[lx-1:rx, ly-1:ry]
m, n = Image.shape

# Define the polynomial degree
degs = [20,30,40,50,60,70,80,90]

# Initialize
CR_VSDK, CR_POLY = [], []
MSE_VSDK, MSE_POLY = [], []
PSNR_VSDK, PSNR_POLY = [], []

# Define the evaluation points
X, Y = np.meshgrid(range(n), range(m));
x = X.T.flatten()
y = Y.T.flatten()
pts = np.vstack((x,y)).T
ptsv = np.vstack((np.array(x/(n-1)), np.array(y/(m-1)))).T

fvalev = np.array([Image[y[i], x[i]] for i in range(x.shape[0])])
threshold = fvalev > 0
extra_ep = np.zeros(Image.flatten().shape)
extra_ep[threshold] = 1
epoints = np.hstack((ptsv, np.matrix(extra_ep).T))

for idx, deg in enumerate(degs):
    print('Testing polynomial of degree %d' %deg)
    t1 = time.time()
    
    # Compute Padova points
    Pad_x, Pad_y = _pdpts(int(np.floor(2 * deg * np.log(deg))))
    pts_x, pts_y = (Pad_x + 1) / 2., (Pad_y + 1) / 2.
    xpts, ypts = pts_x * (n-1), pts_y * (m-1)
    PDpts = np.vstack((xpts.T, ypts.T)).T
    
    # Compute the function values at Padova points
    fval = [Image[int(np.floor(ypts[i])), int(np.floor(xpts[i]))] for i in range(xpts.shape[0])]
    fval = np.matrix(fval).T
    
    # Compute the polynomial approximant
    cfs, lsp = _wamfit(deg, PDpts, pts, fval)
    
    t2 = time.time()
    
    # Compute the nodes for comparisons with VSDKs
    Pad_xv, Pad_yv = _pdpts_vsdk(np.floor(np.sqrt(cfs.shape[0])), cfs.shape[0])
    pts_xv, pts_yv = (Pad_xv+1) / 2, (Pad_yv+1) / 2
    xptsv, yptsv = pts_xv * (n-1), pts_yv * (m-1)
    
    PDptsv = np.matrix(np.vstack((np.array((xptsv/(n-1)).T), np.array((yptsv/(m-1)).T))).T)
    Pti_Mirko = PDptsv.copy() 
    
    fvalv = [Image[int(np.floor(yptsv[i])), int(np.floor(xptsv[i]))] for i in range(xptsv.shape[0])]
    fvalv = np.matrix(fvalv).T
    fvalv_Mirko = fvalv.copy()
    
    threshold = fvalv > 0
    extra_ds = np.zeros(fvalv.shape) 
    extra_ds[threshold] = 1
    dsites = np.hstack((PDptsv, np.matrix(extra_ds)))
    
    # Compute kernel and evaluation matrices
    NN = dsites.shape[0]
    DM = np.zeros((NN, NN))
    NN1 = epoints.shape[0]
    DM_eval = np.zeros((NN1, NN))

    for count in range(3):
        dr, cc = np.meshgrid(epoints[:,count], dsites[:,count])
        DM_eval = DM_eval + (np.power((dr-cc), 2)).T
        dr, cc = np.meshgrid(dsites[:,count], dsites[:,count]);
        DM = DM + (np.power((dr-cc),2)).T
        
    IM1 = _rbfm(1, np.sqrt(DM))
    EM1 = _rbfm(1, np.sqrt(DM_eval))
    
    # Compute and evaluate the VSDKs interpolant
    coef = np.linalg.solve(IM1, fvalv)
    Pf = (EM1.dot(coef))
    
    t3 = time.time()
    
    Imageapprox = np.reshape(lsp, (m,n), order='F')
    Imageapprox1 = np.reshape(Pf, (m,n), order='F')
    
    snr = psnr(Image,Imageapprox); 
    snr1 = psnr(Image,Imageapprox1)
    rmse = mse(Imageapprox,Image); 
    rmse1 = mse(Imageapprox1,Image)
    
    #
    print('MSE for polynomials: %.3e' %rmse)
    print('MSE for VSDKs: %.3e' %rmse1)
    #
    
    # Variograms
    incr = 10
    fvalk, fvalkp, fvalkv, ptk1, ptk2 = [], [], [], [], []
    for i in range(0, pts.shape[0], incr):
        ptk1.append(pts[i, 1])
        ptk2.append(pts[i, 0])
        fvalk.append(fvalev[i])
        fvalkp.append(lsp[i, 0])
        fvalkv.append(Pf[i, 0])

    OK = OrdinaryKriging(ptk1, ptk2, fvalk, variogram_model='spherical', nlags=30)
    OK1 = OrdinaryKriging(ptk1, ptk2, fvalkp, variogram_model='spherical', nlags=30)
    OK2 = OrdinaryKriging(ptk1, ptk2, fvalkv, variogram_model='spherical', nlags=30)
    OK3 = OrdinaryKriging(yptsv, xptsv, fvalv, variogram_model='spherical', nlags=30)
    
    CR_VSDK.append(Pf.shape[0] / coef.shape[0])
    CR_POLY.append(Pf.shape[0] / cfs.shape[0])
    MSE_VSDK.append(rmse1)
    MSE_POLY.append(rmse)
    PSNR_VSDK.append(snr1)
    PSNR_POLY.append(snr)
    
    plot_variogram(OK,  (.0,0.06), 'Sim_variogram_orig_im')
    plot_variogram(OK1, (.0,0.06), 'Sim_variogram_poly_%d' %deg)
    plot_variogram(OK2, (.0,0.06), 'Sim_variogram_VSDK_%d' %deg)
    plot_variogram(OK3, (.0,0.06), 'Sim_variogram_points_%d' %deg)
    
    Outfile = open('Log_SimulatedImage_%d.txt' %deg, "w")
    out_string = 'Partial Sill for original image: %s\n' %(OK.variogram_model_parameters[0])
    out_string += 'Full Sill for original image: %s\n' %(OK.variogram_model_parameters[0]+OK.variogram_model_parameters[2])
    out_string += 'Range for original image: %s\n' %(OK.variogram_model_parameters[1])
    out_string += 'Nugget for original image: %s\n' %(OK.variogram_model_parameters[2])
    out_string += 'Partial Sill for polynomial image: %s\n' %(OK1.variogram_model_parameters[0])
    out_string += 'Full Sill for polynomial image: %s\n' %(OK1.variogram_model_parameters[0]+OK1.variogram_model_parameters[2])
    out_string += 'Range for polynomial image: %s\n' %(OK1.variogram_model_parameters[1])
    out_string += 'Nugget for polynomial image: %s\n' %(OK1.variogram_model_parameters[2])
    out_string += 'Partial Sill for VSDK image: %s\n' %(OK2.variogram_model_parameters[0])
    out_string += 'Full Sill for VSDK image: %s\n' %(OK2.variogram_model_parameters[0]+OK2.variogram_model_parameters[2])
    out_string += 'Range for VSDK image: %s\n' %(OK2.variogram_model_parameters[1])
    out_string += 'Nugget for VSDK image: %s\n' %(OK2.variogram_model_parameters[2])
    out_string += 'Partial Sill for VSDK points: %s\n' %(OK3.variogram_model_parameters[0])
    out_string += 'Full Sill for VSDK points: %s\n' %(OK3.variogram_model_parameters[0]+OK3.variogram_model_parameters[2])
    out_string += 'Range for VSDK points: %s\n' %(OK3.variogram_model_parameters[1])
    out_string += 'Nugget for VSDK points: %s\n' %(OK3.variogram_model_parameters[2])
    out_string += 'Original number of points: %s\n' %(Pf.shape[0])
    out_string += 'Number of Padova points for polynomials: %s\n' %(PDpts.shape[0])
    out_string += 'Number of coefficients for polynomials: %s\n' %(cfs.shape[0])
    out_string += 'Compress ratio for polynomials: %s\n' %(Pf.shape[0]/cfs.shape[0])
    out_string += 'Number of Padova points for VSDKs: %s\n' %(PDptsv.shape[0])
    out_string += 'Number of coefficients for VSDKs: %s\n' %(coef.shape[0])
    out_string += 'Compress ratio for VSDK: %s\n' %(Pf.shape[0]/coef.shape[0])
    out_string += 'CPU time for polynomials: %s\n' %(t2-t1)
    out_string += 'CPU time for VSDKs: %s\n' %(t3-t2)
    out_string += 'PSNR for polynomials: %s\n' %(snr)
    out_string += 'PSNR for VSDKs: %s\n' %(snr1)
    out_string += 'MSE for polynomials: %s\n' %(rmse)
    out_string += 'MSE for VSDKs: %s\n' %(rmse1)
    Outfile.write(out_string)
    Outfile.close()

    # Save data
    np.savetxt('Simulated_square_OriginalImage_xy', pts)
    np.savetxt('Simulated_square_OriginalImage_z', Image)
    np.savetxt('Simulated_ApproximatedImage_PPD_z_%d' %deg, Imageapprox)
    np.savetxt('Simulated_ApproximatedImage_PPD_VSDK_z_%d' %deg, Imageapprox1)
    np.savetxt('Simulated_square_OriginalImage_PPD_xy_%d' %deg, PDpts)
    np.savetxt('Simulated_square_OriginalImage_PPD_z_%d' %deg, fval)
    np.savetxt('Simulated_square_OriginalImage_PPD_xy_VSDK_%d' %deg, PDptsv)
    np.savetxt('Simulated_square_OriginalImage_PPD_z_VSDK_%d' %deg, fval)
    
    # Save images
    plot_image(Imageapprox1, 'Simulated_ApproximatedImage_PPD_VSDK_%d' %deg)
    plot_image(Imageapprox, 'Simulated_ApproximatedImage_PPD_%d' %deg)
    plot_image(Image, 'Simulated_OriginalImage')
    plot_image(Image, 'Simulated_OriginalImage_PPD_%d' %deg, pts=(xptsv,yptsv))

plot_error(CR_VSDK, MSE_VSDK,'CR','MSE','SimulatedCR_VS_MSE_PPD_VSDK')
plot_error(CR_POLY, MSE_POLY,'CR','MSE','SimulatedCR_VS_MSE_PPD')
plot_error(CR_VSDK, PSNR_VSDK,'CR','PSNR','SimulatedCR_VS_PSNR_PPD_VSDK')
plot_error(CR_POLY, PSNR_POLY,'CR','PSNR','SimulatedCR_VS_PSNR_PPD')
