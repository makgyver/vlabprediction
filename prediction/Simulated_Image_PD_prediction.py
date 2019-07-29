# -*- coding: utf-8 -*-

# Import the needed libraries
import scipy.io
import numpy as np
import matplotlib.pylab as plt
from numpy.random import randn
from filterpy.kalman import EnsembleKalmanFilter as EnKF
from filterpy.common import Q_discrete_white_noise
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import load_images_module
from pykrige.ok import OrdinaryKriging

np.random.seed(1234)

# Define the RBF
def _rbfm(ep, r):
    return np.exp(-ep*r)

def hx(x):
    return np.array([x[0]])

def fx(x, dt):
    return np.dot(F, x)

def mse(im1, im2, N):
    return np.sum((np.power(im1-im2, 2))) / N

# Function for computing Padova points
def _pdpts1(n,n1):
    Pad_x = [0]; Pad_x = np.array(Pad_x)
    while Pad_x.shape[0] < n1:
        xyrange = np.array([-1,1,-1,1])
        zn = (xyrange[0]+xyrange[1]+(xyrange[1]-xyrange[0])* \
              np.cos(np.linspace(0,1,n+1)*np.pi))/2;
        zn1 = (xyrange[2]+xyrange[3]+(xyrange[3]-xyrange[2])* \
              np.cos(np.linspace(0,1,n+2)*np.pi))/2;
        Pad1, Pad2 = np.meshgrid(zn,zn1)
        f1 = np.linspace(0,n,n+1)
        f2 = np.linspace(0,n+1,n+2)
        M1, M2 = np.meshgrid(f1,f2)
        h = np.array(np.mod(M1+M2,2))
        g = np.array(np.concatenate(h.T))
        findM = np.argwhere(g)
        Pad_x = np.concatenate(Pad1.T)[findM]
        Pad_y = np.concatenate(Pad2.T)[findM]
        Pad_x = np.matrix(Pad_x)
        Pad_y = np.matrix(Pad_y)
        n+=1
    return Pad_x, Pad_y 

def create_training_and_test(sequence, look_back=10):
    X_tr, y_tr = [], []
    train_size = len(sequence)-1
    tr = sequence[0:train_size]
    for i in range(len(tr) - look_back - 1):
      a = tr[i:(i+look_back)]
      X_tr.append(a)
      y_tr.append(tr[i + look_back])

    X_te, y_te = [], []
    te = sequence[train_size-look_back:]
    for i in range(len(te) - look_back):
      a = te[i:(i+look_back)]
      X_te.append(a)
      y_te.append(te[i + look_back])
      
    return (np.array(X_tr), np.array(y_tr)), (np.array(X_te), np.array(y_te))

def save_image(img, img_name):
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig("%s.png" %img_name,bbox_inches='tight')
    plt.savefig("%s.eps" %img_name,bbox_inches='tight')
    plt.close()

def save_variogram(data, fig_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data.lags, data.semivariance, 'k*')
    plt.ylim((.004,0.12))
    plt.xlabel('Lag distance')
    plt.ylabel('Semivariance')
    plt.savefig("%s.png" %fig_name)
    plt.savefig("%s.eps" %fig_name)


npd = 1891; # number of padova points. 
base_id = 14 # number of images

# Load the images 

imgs = load_images_module.import_img("satur.out", 1365, True)[base_id::15]
tt = len(imgs)


matr = dict()
xpts = dict()
ypts = dict()
fval = np.zeros((npd,tt))
PP = np.zeros((npd,tt))
Ima = np.zeros((206,151))

for i in range(0,tt):
    Image_large = imgs[i]
    sx, sy = Image_large.shape
    lx, rx, ly, ry = int(85), int(290), int(250), int(400)
    Image = Image_large[lx-1:rx,ly-1:ry] 
       
    m, n = Image.shape
    a1 = np.floor(np.sqrt(npd))   
    Pad_x, Pad_y =_pdpts1(a1,npd)
    pts_x, pts_y = (Pad_x+1)/2, (Pad_y+1)/2
    xpts = pts_x*(n)
    ypts = pts_y*(m)
        
    fvala = [Image[int(np.round(ypts[i1]-1)), int(np.round(xpts[i1]-1))] for i1 in range(0, xpts[:].shape[0])]
    fval[:,i] = np.array(fvala).T

Ima = Image

xx = (np.linspace(0,n-1,n))
yy = (np.linspace(0,m-1,m))
X, Y = np.meshgrid(xx,yy)
xxa = X.copy()
xa = np.array([xxa[:,i] for i in range(0,xxa.shape[1])]).flatten()
yya = Y.copy()
ya = np.array([yya[:,i] for i in range(0,yya.shape[1])]).flatten()


#  Kalman filter  
for i in range(1,npd):    
    F = np.array([[1., 0.],[0., 1.]])
    zz = fval[:][i]
    x = np.array([1., 0.])
    P = np.eye(2) * 100.
    enf = EnKF(x=x, P=P, dim_z=1, dt=1., N=5, hx=hx, fx=fx)

    std_noise = 1e-6
    enf.R *= std_noise**2
    enf.Q = Q_discrete_white_noise(2, std_noise, std_noise)
    
    results = np.zeros((1,tt)).T
    for t in range(0,tt):
        # create measurement = t plus white noise
        z = zz[t]+ randn()*std_noise

        enf.predict()
        enf.update(np.asarray([z]))
        # save data
        if zz[t] >  1e-12:
            results[t]=enf.x[0]
        else:
            results[t]=zz[t]
                
        
    results = results.flatten()
    PP[i,:] =  results


#FIT SVM
grid = {"C": [10**e for e in range(-1, 7)], 
        "gamma": [10**e for e in range(-7,-2)], 
        "epsilon": [10**e for e in range(-2,-1)]}
svrs = []
preds = []
for i in range(npd):  
  (X_tr, y_tr), (X_te, y_te) = create_training_and_test(fval[i,:])
  svr = SVR(kernel='rbf', gamma=10**-3, C=1.)#GridSearchCV(SVR(kernel='rbf'), cv=2, param_grid=grid, verbose=True, n_jobs=8)
  svr.fit(X_tr, y_tr)
  svrs.append(svr)
  y_te_pred = svr.predict(X_te)
  print(y_te - y_te_pred)
  preds.append(y_te_pred)

fvalf1 = np.array(preds).flatten()

wam1 = np.matrix(np.vstack((np.array((xpts/sy).T), np.array((ypts/sx).T))).T)
pts = np.matrix(np.vstack((np.array(xa/sy),np.array(ya/sx))).T)

fvalf = np.matrix(fvalf1).T

add_ep1 = Ima.T.flatten()
super_threshold_indices = add_ep1 > 0
add_ep = 0*Image.flatten()
add_ep[super_threshold_indices] = 1

super_threshold_indices = add_ep1 > 0.2
add_ep[super_threshold_indices] = 0.2

super_threshold_indices = add_ep1 > 0.4
add_ep[super_threshold_indices] = 0.4

super_threshold_indices = add_ep1 > 0.6
add_ep[super_threshold_indices] = 0.6

super_threshold_indices = add_ep1 > 0.8
add_ep[super_threshold_indices] = 0.8


add_ep2 = fvalf
super_threshold_indices = add_ep2 > 0
add_ep3 = 0*fvalf
add_ep3[super_threshold_indices] = 1

super_threshold_indices = add_ep2 > 0.2
add_ep3[super_threshold_indices] = 0.2

super_threshold_indices = add_ep2 > 0.4
add_ep3[super_threshold_indices] = 0.4

super_threshold_indices = add_ep2 > 0.6
add_ep3[super_threshold_indices] = 0.6

super_threshold_indices = add_ep2 > 0.8
add_ep3[super_threshold_indices] = 0.8


dsites = np.hstack((wam1,add_ep3))
epoints = np.hstack((pts,np.matrix(add_ep).T))

# Compute kernel and evaluation matrices
DM = np.zeros((dsites.shape[0], dsites.shape[0]))        
DM_eval = np.zeros((epoints.shape[0], dsites.shape[0]))      
for count in range(0,3):
    dr, cc = np.meshgrid(epoints[:,count],dsites[:,count])
    DM_eval = DM_eval + (np.power((dr-cc),2)).T
    dr, cc = np.meshgrid(dsites[:,count],dsites[:,count]); 
    DM = DM + (np.power((dr-cc),2)).T
IM1 = _rbfm(0.1,np.sqrt(DM))
EM1 = _rbfm(0.1,np.sqrt(DM_eval))

# Compute and evaluate the VSDKs interpolant
coef = np.linalg.solve(IM1,fvalf)
Pf = (EM1.dot(coef))
Imageapprox1 = np.reshape(Pf,(m,n),order='F')


fvalf2 = PP[:,-1]
fvalf3 = np.matrix(fvalf2).T

add_ep2 = fvalf3
super_threshold_indices = add_ep2 > 0
add_ep3 = 0*fvalf
add_ep3[super_threshold_indices] = 1

super_threshold_indices = add_ep2 > 0.2
add_ep3[super_threshold_indices] = 0.2

super_threshold_indices = add_ep2 > 0.4
add_ep3[super_threshold_indices] = 0.4

super_threshold_indices = add_ep2 > 0.6
add_ep3[super_threshold_indices] = 0.6

super_threshold_indices = add_ep2 > 0.8
add_ep3[super_threshold_indices] = 0.8

# Compute kernel and evaluation matrices
DM = np.zeros((dsites.shape[0], dsites.shape[0]))        
DM_eval = np.zeros((epoints.shape[0], dsites.shape[0]))      
for count in range(0,3):
    dr, cc = np.meshgrid(epoints[:,count],dsites[:,count])
    DM_eval = DM_eval + (np.power((dr-cc),2)).T
    dr, cc = np.meshgrid(dsites[:,count],dsites[:,count]); 
    DM = DM + (np.power((dr-cc),2)).T
IM1 = _rbfm(1,np.sqrt(DM))
EM1 = _rbfm(1,np.sqrt(DM_eval))

# Compute and evaluate the VSDKs interpolant
coef = np.linalg.solve(IM1,fvalf)
Pf2 = (EM1.dot(coef))
Imageapprox2 = np.reshape(Pf2,(m,n),order='F')


rmse1 = mse(Imageapprox1, Image, (Pf.shape[0]))
rmse2 = mse(Imageapprox2, Image, (Pf.shape[0]))

Outfile = open('Prediction_results._%d.txt'%npd,"w")
out_string = 'RMSE with SVM: %s\n' %(rmse1)
out_string += 'RMSE with EnKF: %s\n' %(rmse2)
Outfile.write(out_string)   
Outfile.close()

save_image(Imageapprox1, "fig_svm_%d" %npd)
save_image(Imageapprox2, "fig_enkf_%d" %npd)
save_image(Image, "original_p")

pts = np.matrix(np.vstack((np.array(xa), np.array(ya))).T)

# Variograms 
incr = 10; fvalk = [];  fvalkv = []; fvalv = []; ptk1 = []; ptk2 = []

X, Y = np.meshgrid(range(n), range(m));
x, y = X.T.flatten(), Y.T.flatten()
y = fvalev = np.array([Image[y[i], x[i]] for i in range(x.shape[0])])

for i in range(0, pts.shape[0], incr):
    ptk1.append(pts[i,0])
    ptk2.append(pts[i,1])
    fvalk.append(fvalev[i])
    fvalkv.append(Pf2[int(i),0])
    fvalv.append(Pf[int(i),0])

# Write the outputs
OK  = OrdinaryKriging(ptk1, ptk2, fvalk, variogram_model='spherical', nlags=30) 
OK2 = OrdinaryKriging(ptk1, ptk2, fvalkv, variogram_model='spherical', nlags=30)
OK3 = OrdinaryKriging(ptk1, ptk2, fvalv, variogram_model='spherical', nlags=30)
 
save_variogram(OK, "Sim_variogram_orig_im_p")
save_variogram(OK2, 'Sim_variogram_ekf_p_%d' %npd)
save_variogram(OK3, 'Sim_variogram_svm_p_%d' %npd)
