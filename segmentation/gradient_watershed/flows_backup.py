# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 03:04:09 2023

@author: fyz11

Module for computing and implementing various flow definitions. 

"""

import numpy as np 


"""
Flow 1 - euclidean distance transform. 
"""
# def distance_transform_labels(labels, bg_label=0):
    
#     import numpy as np 
#     from scipy.ndimage import distance_transform_edt
#     # import skfmm
    
#     dtform = np.zeros(labels.shape)
#     dtform_flow = np.zeros((2,)+labels.shape) # keep these separate!. 
    
#     uniq_labels = np.setdiff1d(np.unique(labels), bg_label)
    
#     for lab in uniq_labels:
        
#         mask = labels == lab
#         # dist_mask = skfmm.distance(mask>0)
#         # dist_mask = distance_transform_edt(mask>0)
#         dist_mask = sdf_distance_transform(mask>0)
        
#         # compute the gradient!. 
#         grad_dist_mask = np.array(np.gradient(dist_mask))
#         grad_dist_mask = grad_dist_mask / (np.linalg.norm(grad_dist_mask, axis=0)[None,...] + 1e-20)
        
#         dtform_flow[:,mask>0] = grad_dist_mask[:,mask>0].copy()
        
#         dtform[mask>0] = dist_mask[mask>0]
        
#     return dtform_flow, dtform 


# =============================================================================
#  Use the euclidean distance transform given in the edt library which is faster.
# =============================================================================
def distance_transform_labels_fast(labels, sdf=False, n_threads=16, black_border=False):
    """ compute euclidean distance transform for each uniquely labelled cell in a 2D/3D binary image 
    
    """
    import numpy as np 
    import edt # external fast distance transform from seung lab. https://github.com/seung-lab/euclidean-distance-transform-3d # can also compute sdf?
    
    if not sdf:
        
        if len(labels.shape) == 2:
            dtform = edt.edt(labels, 
                             black_border=black_border, 
                             order='C', 
                             parallel=n_threads).astype(np.float32)    
        else:
            # does this not work for a 3D label set?
            dtform = np.array([edt.edt(ss, black_border=black_border, 
                                       order='C', 
                                       parallel=n_threads).astype(np.float32) for ss in labels])
    else:
        if len(labels.shape) == 2:
            dtform = edt.sdf(labels, black_border=black_border,
                             order='C', 
                             parallel=n_threads).astype(np.float32)
        else:
            dtform = np.array([edt.sdf(ss, black_border=black_border, 
                                       order='C', 
                                       parallel=n_threads).astype(np.float32) for ss in labels])
        
    return dtform 



# =============================================================================
#   Use the poisson heat distance transform 
# =============================================================================

# construct the laplacian grid for a 2D image. (might be faster to exploit kron! )
def _laplacian_matrix(n, m, mask=None): # this is the only working solution! 
    
    import scipy.sparse
    from scipy.sparse.linalg import spsolve
    
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
    
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    if mask is not None:
        y_range = mask.shape[0]
        x_range = mask.shape[1]
        # find the masked i.e. zeros
        zeros = np.argwhere(mask==0) # in (y,x)
        k = zeros[:,1] + zeros[:,0] * x_range
        mat_A[k,k] = 1
        mat_A[k, k + 1] = 0
        mat_A[k, k - 1] = 0
        mat_A[k, k + x_range] = 0
        mat_A[k, k - x_range] = 0
        
        mat_A = mat_A.tocsc()
    else:
        mat_A = mat_A.tocsc()
    
    return mat_A

def poisson_dist_tform(binary, pt=None):
    """
    Computation for a single binary image. 
    """
    import scipy.sparse
    from scipy.sparse.linalg import spsolve
    import numpy as np 
    
    mask = np.pad(binary, [1,1], mode='constant', constant_values=1) # pad with ones.  # ones need for connectivity....
    
    mat_A = _laplacian_matrix(mask.shape[0], mask.shape[1], mask=mask) # this is correct!

    if pt is not None:
        mask = np.zeros_like(mask)
        mask[pt[0], pt[1]] = 1
        
        mask_flat = mask.flatten()    
        mat_b = np.zeros(len(mask_flat))
        mat_b[mask_flat == 1] = 1
        
    else:
        # now we need to zero the padding
        mask[0,:]=0
        mask[:,-1]=0
        mask[-1,:]=0
        mask[:,0]=0
        
        # solve within the mask!    
        mask_flat = mask.flatten()    
        # inside the mask:
        # \Delta f = div v = \Delta g       
        mat_b = np.ones(len(mask_flat)) # does this matter -> the amount... 
        mat_b[mask_flat == 0] = 0
        
    x = spsolve(mat_A, mat_b)    
    x = x.reshape(mask.shape)
    x = x[1:-1,1:-1].copy()
    x = x - x.min() # solution is only positive!.
    return x 
    
def poisson_dist_tform_flow(coords, centroid, shape, power_dist=None):
    """
    Solve diffusion for points. 
    """
    import numpy as np 
    
    mask = np.zeros(shape, dtype=np.bool)
    mask[coords[:,0], coords[:,1]] = 1
    
    poisson = poisson_dist_tform(mask>0, pt=centroid) # compute the point source distance transform 

    # if log_dist:
        # poisson = np.log(poisson+1.)
    if power_dist is not None:
        # why are we getting nan? wiht power boost? 
        # this is to boost beyond the numerical limits. 
        poisson = np.clip(poisson, 0, 1)
        poisson = poisson**power_dist # valid for 0-1
        
    poisson_flow = np.array(np.gradient(poisson))

    dy = poisson_flow[0, coords[:,0], coords[:,1]].copy()
    dx = poisson_flow[1, coords[:,0], coords[:,1]].copy()
    
    return poisson[coords[:,0], coords[:,1]], np.stack((dy,dx))


# =============================================================================
#   Defining the fmm geodesic flow
# =============================================================================
# def fmm_point_source2D(binary, pt):
#     """
#     This is a much better solver of the diffusion
#     """
    
#     import skfmm
#     import numpy as np
#     import skimage.morphology as skmorph
#     import scipy.ndimage as ndimage 
    
#     # create a masked array
#     # mask = np.logical_not(skmorph.binary_dilation(binary, skmorph.disk(1)))
#     mask = np.logical_not(binary).copy()
#     # mask2 = np.logical_not(skmorph.binary_dilation(binary, skmorph.disk(1)))
#     mask2 = mask.copy()
#     m = np.ones_like(binary)
#     m[pt[0],pt[1]] = 0
#     m = np.ma.masked_array(m, mask2)

#     dist_image = skfmm.distance(m)  # this is the answer!!!     
#     dist_image = dist_image.max()-dist_image # invert the image!. # many ways to do this. 
    
#     # fix up the boundary differentiation. 
#     dist_outer = (skfmm.distance(mask)*-1) # can we make it smoother? # best
#     dist_image[mask>0] = dist_outer[mask>0] # this seems to work! (it gives a little normal push. ) 
#     # dist_image = dist_image/dist_image.max() # retain this. 
    
#     dist_gradient = np.array(np.gradient(dist_image))
#     # dist_gradient = dist_gradient/(np.linalg.norm(dist_gradient, axis=0)[None,...]+1e-20) # normalise. 
#     dist_gradient[:,mask>0] = 0
#     dist_image[mask>0]=0
    
#     return dist_image, dist_gradient

def fmm_point_source2D(coords, centroid, shape):
    """
    This is a much better solver of the diffusion
    """
    
    import skfmm
    import numpy as np
    import skimage.morphology as skmorph
    import scipy.ndimage as ndimage 
    
    # construct the binary 
    binary = np.zeros(shape, dtype=np.bool)
    binary[coords[:,0], coords[:,1]] = 1
    
    # construct the masked out region!. 
    mask = np.logical_not(binary).copy()
    # mask2 = np.logical_not(skmorph.binary_dilation(binary, skmorph.disk(1)))
    mask2 = mask.copy()
    m = np.ones_like(binary)
    m[centroid[0],centroid[1]] = 0 # define the point source!. 
    m = np.ma.masked_array(m, mask2)

    dist_image = skfmm.distance(m)  # this is the answer!!!     
    dist_image = dist_image.max()-dist_image # invert the image!. # many ways to do this. 
    
    # fix up the boundary differentiation. 
    dist_outer = (skfmm.distance(mask)*-1) # can we make it smoother? # best
    dist_image[mask>0] = dist_outer[mask>0] # this seems to work! (it gives a little normal push. ) 
    # dist_image = dist_image/dist_image.max() # retain this. 
    
    dist_gradient = np.array(np.gradient(dist_image))
    
    dy = dist_gradient[0, coords[:,0], coords[:,1]].copy()
    dx = dist_gradient[1, coords[:,0], coords[:,1]].copy()
    
    return dist_image[coords[:,0], coords[:,1]], np.stack((dy,dx))
    
   
# =============================================================================
#   Replicating cellpose flow
# =============================================================================
from numba import njit, float32, int32, vectorize
@njit('(float64[:], int32[:], int32[:], int32, int32, int32, int32)', nogil=True)
def _extend_centers(T,y,x,ymed,xmed,Lx, niter):
    """ run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)
    
    taken from  cellpose source code.
    
    Parameters
    --------------
    T: float64, array
        _ x Lx array that diffusion is run in
    y: int32, array
        pixels in y inside mask
    x: int32, array
        pixels in x inside mask
    ymed: int32
        center of mask in y
    xmed: int32
        center of mask in x
    Lx: int32
        size of x-dimension of masks
    niter: int32
        number of iterations to run diffusion
    Returns
    ---------------
    T: float64, array
        amount of diffused particles at each pixel
    """

    for t in range(niter):
        T[ymed*Lx + xmed] += 1
        T[y*Lx + x] = 1/9. * (T[y*Lx + x] + T[(y-1)*Lx + x]   + T[(y+1)*Lx + x] +
                                            T[y*Lx + x-1]     + T[y*Lx + x+1] +
                                            T[(y-1)*Lx + x-1] + T[(y-1)*Lx + x+1] +
                                            T[(y+1)*Lx + x-1] + T[(y+1)*Lx + x+1])
    return T


def cellpose_diffuse2D(coords, centroid, shape, niter):
    """
    Solve diffusion for points. 
    """
    import numpy as np 
    
    ly, lx = shape
    
    x = coords[:,1].copy() # these are the diffusion points to solve for. 
    y = coords[:,0].copy()

    T = np.zeros((ly+2)*(lx+2), np.float64)
    T = _extend_centers(T,
                        y=y.astype(np.int32),
                        x=x.astype(np.int32),
                        ymed=centroid[0].astype(np.int32),
                        xmed=centroid[1].astype(np.int32),
                        Lx=int32(lx), 
                        niter=int32(niter))
    T[(y+1)*lx + x+1] = np.log(1.+T[(y+1)*lx + x+1])
    dy = T[(y+1)*lx + x] - T[(y-1)*lx + x]
    dx = T[y*lx + x+1] - T[y*lx + x-1]
    
    return T[(y+1)*lx + x+1], np.stack((dy,dx))


# =============================================================================
#   Generic wrapper functions.  
# =============================================================================

def distance_tform_labels2D(labelled, dtform_fnc, clip=False): 
    """
    this function wraps a distance transform defined for a single binary and generalize to multi-labels. 
    """
    import skimage.measure as skmeasure
    import numpy as np 
    
    distance = np.zeros(labelled.shape)
    binary = labelled>0
    
    labelled_regions = skmeasure.regionprops(labelled)
    
    for reg in labelled_regions:
        
        patch = reg.image
        box = reg.bbox # (min_row, min_col, max_row, max_col)
        y1, x1, y2, x2 = box
        coords = reg.coords
        
        dist = dtform_fnc(patch) 
        
        if clip:
            dist = dist / dist.max()
            dist = np.clip(dist, 0, 1)
        else:
            dist = np.clip(dist, 0, np.inf)
        
        # copy it into the larger array. 
        distance[coords[:,0], coords[:,1]] = dist[coords[:,0]-y1, coords[:,1]-x1].copy()
        
    return distance


def distance_centroid_tform_flow_labels2D(labelled, dtform_method='cellpose', guide_img=None, fixed_point_percentile=0.9, iter_factor=5, power_dist=None):
    
    """
    This function wraps a distance transform defined for a single binary with point source and generalize to multi-labels. 
    """
    # is there a better way to smooth this? 
    import skimage.measure as skmeasure
    from skimage.morphology import skeletonize
    import skimage.segmentation as sksegmentation 
    import numpy as np 

    distance = np.zeros(labelled.shape)
    flow = np.zeros((2,)+labelled.shape)
    # binary = labelled>0 # to tell us which is not background 
    
    labelled_regions = skmeasure.regionprops(labelled)
    
    for reg in labelled_regions:
        patch = reg.image
        box = reg.bbox # (min_row, min_col, max_row, max_col) # same behavior
        y1, x1, y2, x2 = box
        start = np.hstack([y1,x1])
        coords = reg.coords 
        
        # determine the median 
        centroid = np.nanmedian(coords, axis=0)
        
        
        """
        stability for gauss-siedel methods. 
        """
        inner = sksegmentation.find_boundaries(patch*1, mode='inner')
        inner = np.logical_and(patch, np.logical_not(inner))
        inner_coords = np.argwhere(inner>0)
        
        if len(inner_coords)>0:
            inner_coords = (inner_coords+start[None,:]).copy()
        else:
            inner_coords = coords.copy()
        
        if guide_img is not None:
            # # use this to find an internal fixed point. 
            coords_guide = guide_img[inner_coords[:,0],inner_coords[:,1]].copy()
            coords_valid = inner_coords[coords_guide>=np.percentile(coords_guide, fixed_point_percentile*100)]
            # coords_valid = np.argwhere(skeletonize(patch>0)>0)+start[None,:]
            centroid = coords_valid[np.argmin(np.linalg.norm(coords_valid-centroid[None,:], axis=1))]
        else:
            centroid = inner_coords[np.argmin(np.linalg.norm(inner_coords-centroid[None,:], axis=1))]
            
            
        if dtform_method == 'cellpose':
            niter = iter_factor*np.int32(np.ptp(coords[:,1]) + np.ptp(coords[:,0])) # there should be a better estimate of this. 
            
            dist_, flow_ = cellpose_diffuse2D(coords-start[None,:]+1, # remove the global offset. 
                                           centroid-start+1,
                                            shape=np.hstack(patch.shape)+1, 
                                            niter=niter)
        
        if dtform_method == 'cellpose_improve':    
            dist_, flow_ = poisson_dist_tform_flow(coords-start[None,:]+1, # remove the global offset. 
                                                   centroid-start+1,
                                                   shape=np.hstack(patch.shape)+1,
                                                   power_dist=power_dist)
        
        if dtform_method == 'fmm':  
            # geodesic fast marching method
            dist_, flow_ = fmm_point_source2D(coords-start[None,:]+1, # remove the global offset. 
                                                   centroid-start+1,
                                                   shape=np.hstack(patch.shape)+1)
        
        
        # diffuse_out = out.reshape(ly+2,lx+2, order='')
        distance[coords[:,0],coords[:,1]] = dist_.copy()
        flow[:,coords[:,0],coords[:,1]] = flow_.copy()
        
    # normalise
    flow /= (1e-20 + (flow**2).sum(axis=0)**0.5)
        
    return np.concatenate([distance[None,...], flow], axis=0) # concatenate this!.  # then we compute both!. 


# =============================================================================
# Parallel scripts
# =============================================================================

def _distance_centroid_tform_flow_labels2D_chunk(labelled_array, 
                                                 dtform_method='cellpose', 
                                                 guide_img=None,
                                                 fixed_point_percentile=0.9,
                                                 iter_factor=5,
                                                 power_dist=None):

     import numpy as np

     if guide_img is not None:
         dtform_flow = np.array([distance_centroid_tform_flow_labels2D(labelled_array[zz], 
                                                                       dtform_method = dtform_method,
                                                                       guide_img=guide_img[zz], 
                                                                       fixed_point_percentile=fixed_point_percentile,
                                                                       iter_factor=iter_factor,
                                                                       power_dist=power_dist) for zz in np.arange(len(labelled_array))])
     else:
         dtform_flow = np.array([distance_centroid_tform_flow_labels2D(labelled_array[zz], 
                                                                       dtform_method = dtform_method,
                                                                       guide_img=None, 
                                                                       fixed_point_percentile=fixed_point_percentile,
                                                                       iter_factor=iter_factor,
                                                                       power_dist=power_dist) for zz in np.arange(len(labelled_array))])
     return dtform_flow
 
    
def distance_centroid_tform_flow_labels2D_parallel(labelled_array, 
                                                   dtform_method='cellpose',
                                                    guide_image=None,
                                                    fixed_point_percentile=0.9,
                                                    iter_factor=5, 
                                                    n_processes=4,
                                                    power_dist=None):
    import multiprocessing as mp
    import numpy as np 
    
    n_proc = n_processes
    chunksize = len(labelled_array)//n_proc
    
    # chunk the data. 
    chunks = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None
        chunks.append(labelled_array[slice(chunkstart, chunkend)])
 
    assert sum(map(len, chunks)) == len(labelled_array)
         
    # parallel computing 
    with mp.Pool(processes=n_proc) as pool:
        # starts the sub-processes without blocking
        # pass the chunk to each worker process
        # proc_results = [pool.apply_async(centroid_geodesic_distance_transform2D_chunk,
        #                                    args=(chunk,))
        #                                  for chunk in chunks]
        if dtform_method == 'cellpose_improve':
            proc_results = [pool.apply_async(_distance_centroid_tform_flow_labels2D_chunk,
                                               args=(chunk, dtform_method, guide_image,fixed_point_percentile, power_dist))
                                             for chunk in chunks]
        else:
            proc_results = [pool.apply_async(_distance_centroid_tform_flow_labels2D_chunk,
                                               args=(chunk, dtform_method, guide_image,fixed_point_percentile, iter_factor))
                                             for chunk in chunks]
         
        # blocks until all results are fetched
        res = np.vstack([r.get() for r in proc_results])
         
    return res


# parallel above but for dtforms
def _distance_tform_labels2D_chunk(labelled_array, 
                                    dtform_fnc, clip=False):

     import numpy as np
     
     dtform = np.array([distance_tform_labels2D(labelled_array[zz], 
                                                dtform_fnc = dtform_fnc,
                                                clip=clip) for zz in np.arange(len(labelled_array))])
     
     return dtform
 
    
def distance_tform_labels2D_parallel(labelled_array, 
                                     dtform_fnc,
                                     clip=False, 
                                     n_processes=4):
    import multiprocessing as mp
    import numpy as np 
    
    n_proc = n_processes
    chunksize = len(labelled_array)//n_proc
    
    # chunk the data. 
    chunks = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None
        chunks.append(labelled_array[slice(chunkstart, chunkend)])
 
    assert sum(map(len, chunks)) == len(labelled_array)
         
    # parallel computing 
    with mp.Pool(processes=n_proc) as pool:
        # starts the sub-processes without blocking
        proc_results = [pool.apply_async(_distance_tform_labels2D_chunk,
                                           args=(chunk, dtform_fnc, clip))
                                         for chunk in chunks]
       
        # blocks until all results are fetched
        res = np.vstack([r.get() for r in proc_results])
         
    return res



def _relabel_slices(labelled, bg_label=0):
    
    import numpy as np 
    max_ID = 0 
    labelled_ = []
    for lab in labelled:
        lab[lab>bg_label] = lab[lab>bg_label] + max_ID # only update the foreground! 
        labelled_.append(lab)
        max_ID = np.max(lab)+1
        
    labelled_ = np.array(labelled_)
    
    return labelled_


