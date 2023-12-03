# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 22:35:23 2023

@author: fyz11
"""

def _mkdir(directory):
    
    import os 
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    return []

def _normalize99(Y,lower=0.01,upper=99.99):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile 
    Upper and lower percentile ranges configurable. 
    
    Parameters
    ----------
    Y: ndarray, float
        Component array of lenth N by L1 by L2 by ... by LN. 
    upper: float
        upper percentile above which pixels are sent to 1.0
    
    lower: float
        lower percentile below which pixels are sent to 0.0
    
    Returns
    --------------
    normalized array with a minimum of 0 and maximum of 1
    
    """
    import numpy as np
    
    X = Y.copy()
    
    return np.interp(X, (np.percentile(X, lower), np.percentile(X, upper)), (0, 1))


def _interp2(query_pts, grid_shape, I_ref, method='linear', cast_uint8=False):
    
    import numpy as np 
    from scipy.interpolate import RegularGridInterpolator 
    
    spl = RegularGridInterpolator((np.arange(grid_shape[0]), 
                                   np.arange(grid_shape[1])), 
                                   I_ref, method=method, bounds_error=False, fill_value=0)
    I_query = spl((query_pts[...,0], 
                   query_pts[...,1]))

    if cast_uint8:
        I_query = np.uint8(I_query)
    
    return I_query
    
def _interp3(query_pts, grid_shape, I_ref, method='linear', cast_uint8=False):
    
    from scipy.interpolate import RegularGridInterpolator
    import numpy as np 
    
    spl_3 = RegularGridInterpolator((np.arange(grid_shape[0]), 
                                         np.arange(grid_shape[1]), 
                                         np.arange(grid_shape[2])), 
                                         I_ref, method=method, bounds_error=False, fill_value=0)
    
    I_query = spl_3((query_pts[...,0], 
                      query_pts[...,1],
                      query_pts[...,2]))
    if cast_uint8:
        I_query = np.uint8(I_query)
    
    return I_query

# =============================================================================
# 2D stuff 
# =============================================================================
def connected_components_pts_2D( pts, pts0, shape, 
                                smooth_sigma=1, 
                                thresh_factor=None, 
                                mask=None,
                                min_area=1) : 

    import numpy as np 
    import scipy.ndimage as ndimage
    import skimage.measure as skmeasure
    import skimage.segmentation as sksegmentation 
    
    # parse ... 
    votes_grid_acc = np.zeros(shape)
    
    # count
    votes_grid_acc[(pts[:,0]).astype(np.int), 
                   (pts[:,1]).astype(np.int)] += 1. # add a vote. 
                   
    # smooth to get a density (fast KDE estimation)
    votes_grid_acc = ndimage.gaussian_filter(votes_grid_acc, sigma=smooth_sigma)  
    
    if thresh_factor is not None:
        if mask is not None:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc[mask]) + thresh_factor*np.std(votes_grid_acc[mask])
        else:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc) + thresh_factor*np.std(votes_grid_acc)
    else:
        votes_grid_binary = votes_grid_acc > np.mean(votes_grid_acc) # just threshold over the mean. 
        
    cell_seg_connected = skmeasure.label(votes_grid_binary, connectivity=1) # use the full conditional 
    cell_uniq_regions = np.setdiff1d(np.unique(cell_seg_connected),0)
    if len(cell_uniq_regions)>0:
        props = skmeasure.regionprops(cell_seg_connected)
        areas = np.hstack([re.area for re in props])
        invalid_areas = cell_uniq_regions[areas<=min_area]
    
        for invalid in invalid_areas:
            cell_seg_connected[cell_seg_connected==invalid] = 0
        
    if cell_seg_connected.max() > 0:
        cell_seg_connected = sksegmentation.relabel_sequential(cell_seg_connected)[0]
    
    
    cell_seg_connected_original = np.zeros_like(cell_seg_connected)
    cell_seg_connected_original[(pts0[:,0]).astype(np.int), 
                                (pts0[:,1]).astype(np.int)] = cell_seg_connected[(pts[:,0]).astype(np.int), 
                                                                                 (pts[:,1]).astype(np.int)]
    
    if mask is not None:
        cell_seg_connected[mask == 0] = 0
        cell_seg_connected_original[mask==0] = 0 # also mask the predicted. 

    return cell_seg_connected_original, cell_seg_connected, votes_grid_acc # return the accumulator.!    


def connected_components_pts_3D( pts, pts0, shape, 
                                smooth_sigma=1, 
                                thresh_factor=None, 
                                mask=None,
                                min_area=1) : 

    import numpy as np 
    import scipy.ndimage as ndimage
    import skimage.measure as skmeasure
    import skimage.segmentation as sksegmentation 
    
    # parse ... 
    votes_grid_acc = np.zeros(shape)
    
    # count
    votes_grid_acc[(pts[:,0]).astype(np.int), 
                   (pts[:,1]).astype(np.int),
                   (pts[:,2]).astype(np.int)] += 1. # add a vote. 
                   
    # smooth to get a density (fast KDE estimation)
    votes_grid_acc = ndimage.gaussian_filter(votes_grid_acc, sigma=smooth_sigma)  
    
    if thresh_factor is not None:
        if mask is not None:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc[mask]) + thresh_factor*np.std(votes_grid_acc[mask])
        else:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc) + thresh_factor*np.std(votes_grid_acc)
    else:
        votes_grid_binary = votes_grid_acc > np.mean(votes_grid_acc) # just threshold over the mean. 
        
    cell_seg_connected = skmeasure.label(votes_grid_binary, connectivity=2) # use the full conditional 
    cell_uniq_regions = np.setdiff1d(np.unique(cell_seg_connected),0)
    if len(cell_uniq_regions)>0:
        props = skmeasure.regionprops(cell_seg_connected)
        areas = np.hstack([re.area for re in props])
        invalid_areas = cell_uniq_regions[areas<=min_area]
    
        for invalid in invalid_areas:
            cell_seg_connected[cell_seg_connected==invalid] = 0
        
    if cell_seg_connected.max() > 0:
        cell_seg_connected = sksegmentation.relabel_sequential(cell_seg_connected)[0]
    
    
    cell_seg_connected_original = np.zeros_like(cell_seg_connected)
    cell_seg_connected_original[(pts0[:,0]).astype(np.int), 
                                (pts0[:,1]).astype(np.int),
                                (pts0[:,2]).astype(np.int)] = cell_seg_connected[(pts[:,0]).astype(np.int), 
                                                                                 (pts[:,1]).astype(np.int),
                                                                                 (pts[:,2]).astype(np.int)]
    
    if mask is not None:
        cell_seg_connected[mask == 0] = 0
        cell_seg_connected_original[mask==0] = 0 # also mask the predicted. 

    return cell_seg_connected_original, cell_seg_connected, votes_grid_acc # return the accumulator.!    


def _sdf_distance_transform(binary, rev_sign=True): 
    
    import numpy as np 
    from scipy.ndimage import distance_transform_edt
    # import skfmm
    # import GeodisTK
    
    pos_binary = binary.copy()
    neg_binary = np.logical_not(pos_binary)
    
    res = distance_transform_edt(neg_binary) * neg_binary - (distance_transform_edt(pos_binary) - 1) * pos_binary
    # res = skfmm.distance(neg_binary, dx=0.5) * neg_binary - (skfmm.distance(pos_binary, dx=0.5) - 1) * pos_binary
    # res = skfmm.distance(neg_binary) * neg_binary - (skfmm.distance(pos_binary) - 1) * pos_binary # this was fast!. 
    # res = geodesic_distance_2d((neg_binary*1.).astype(np.float32), S=neg_binary, lamb=0.8, iter=10) * neg_binary - (geodesic_distance_2d((pos_binary*1.).astype(np.float32), S=neg_binary, lamb=0.5, iter=10) - 1) * pos_binary
    
    if rev_sign:
        res = res * -1
    
    return res



def surf_normal_sdf(binary, return_sdf=True, smooth_gradient=None, eps=1e-12, norm_vectors=True):

    import numpy as np 
    import scipy.ndimage as ndimage

    sdf_vol = _sdf_distance_transform(binary, rev_sign=True) # so that we have it pointing outwards!. 
    
    # compute surface normal of the signed distance function. 
    sdf_vol_normal = np.array(np.gradient(sdf_vol))
    # smooth gradient
    if smooth_gradient is not None: # smoothing needs to be done before normalization of magnitude. 
        sdf_vol_normal = np.array([ndimage.gaussian_filter(sdf, sigma=smooth_gradient) for sdf in sdf_vol_normal])

    if norm_vectors:
        sdf_vol_normal = sdf_vol_normal / (np.linalg.norm(sdf_vol_normal, axis=0)[None,:]+eps)

    return sdf_vol_normal, sdf_vol


def mean_curvature_sdf(sdf_normal):

    def divergence(f):
        import numpy as np 
        """
        Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
        :param f: List of ndarrays, where every item of the list is one dimension of the vector field
        :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
        """
        num_dims = len(f)
        return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])
        
    H = .5*(divergence(sdf_normal))# total curvature is the divergence of the normal. 
    
    return H 


def gradient_watershed2D_binary(binary, 
                                gradient_img=None, 
                                divergence_rescale=True, 
                                smooth_sigma=1, 
                                smooth_gradient=1, 
                                delta=.5, 
                                n_iter=10, 
                                min_area=5, 
                                eps=1e-20, 
                                interp=True,
                                thresh_factor=None, 
                                track_flow=True, # if track_flow then we record!. 
                                mask=None,
                                debug_viz=False):
    
    """ parses the instance level segmentation implicitly given as an input binary or a vector field. 
    The algorithm works as an inverse watershed.
    
    Step 1: a grid of points is seeds on the image
    Step 2: points are propagated for n_iter according to the gradient_img, condensing towards cell centers implicitly implied by the gradient image.
    Step 3: individual cluster centers are found by binarisation and connected component, removing objects < min_area
    
    result is an integer image the same size as binary. 

    Parameters
    ----------
    binary : (MxNxL) numpy array
        input binary image defining the voxels that need labeling
    gradient_img :  (MxNxLx3) numpy array
        This is a gradient field such as that from applying np.array(np.gradient(img)).transpose(1,2,3,0) where img is a potential such as a distance transform or probability map. 
    divergence_rescale : 
        If True, the gradient_img is scaled by the divergence which is equivalent to the mean curvature, this helps to prevent early breakage for tube-like structures.   
    smooth_sigma : scalar
        controls the catchment area for identifying distinct cells at the final propagation position. Smaller smooth_sigma leads to more oversegmentation. 
    smooth_gradient : scalar
        the isotropic sigma value controlling the Gaussian smoothing of the gradient field. More smoothing results in more cells grouped together
    delta: scalar
        the voxel size to propagate grid points per iteration. Related to the stability. If too small takes too long. If too large, might not converge. if delta=1, takes a 1 voxel step. 
    n_iter: int 
        the number of iterations to run. (To do: monitor convergence and break early to improve speed)
    min_area: scalar
        volume of cells < min_area are removed. 
    eps: float
        a small number for numerical stability
    thresh_factor: scalar
        The final cells are identified by thresholding on a threshold mean+thresh_factor*std. Thresh_factor controls what is an object prior to connected components analysis 
    mask: (MxNxL) numpy array
        optional binary mask to gate the region to parse labels for.
    debug_viz: bool
        if True, visualise the position of the points at every algorithm iteration. 
        
    Returns
    -------
    cell_seg_connected_original : (MxNxL)
        an integer image where each unique int > 0 relates to a unique object such that object 1 is retrieved by cell_seg_connected_original==1.
        
    """
    import scipy.ndimage as ndimage
    import numpy as np 
    import skimage.morphology as skmorph
    import pylab as plt 
    import skimage.measure as skmeasure 
    import skimage.segmentation as sksegmentation 
    from tqdm import tqdm 
    
    # compute the signed distance transform
    if gradient_img is not None:
        sdf_normals = gradient_img.transpose(2,0,1) # use the supplied gradients! 
        sdf_normals = sdf_normals * binary[None,...]
    else:
        sdf_normals, sdf_binary = surf_normal_sdf(binary, return_sdf=True, smooth_gradient=smooth_gradient, eps=eps, norm_vectors=True)
        sdf_normals = sdf_normals * binary[None,...]
        
    if divergence_rescale:
        # rescale the speed
        curvature_2D = mean_curvature_sdf(sdf_normals/(np.linalg.norm(sdf_normals, axis=0)[None,...]+eps))
        curvature_2D = _normalize99(curvature_2D) # rescales to a factor between 0-1
        sdf_normals = sdf_normals * curvature_2D[None,...] # multiplicative factor rescaling 
        
        
    # print(sdf_normals.shape)
    grid =  np.zeros(binary.shape, dtype=np.int32)
    pts = np.argwhere(binary>0) # (N,ndim)
    
    tracks = [pts]
    
    for ii in tqdm(np.arange(n_iter)):
        pt_ii = tracks[-1].copy()
        
        if interp:
            pts_vect_ii = np.array([_interp2(pt_ii, binary.shape, I_ref=sdf_normals[ch], method='linear', cast_uint8=False) for ch in np.arange(len(sdf_normals))]).T
        else:
            pts_vect_ii = sdf_normals[:,np.rint(pt_ii[:,0]).astype(np.int64), np.rint(pt_ii[:,1]).astype(np.int64)].T
        
        pts_vect_ii = pts_vect_ii / (np.linalg.norm(pts_vect_ii, axis=-1)[:,None] + eps)
        pt_ii_next = pt_ii + delta*pts_vect_ii
            
        pt_ii_next[:,0] = np.clip(pt_ii_next[:,0], 0, binary.shape[0]-1)
        pt_ii_next[:,1] = np.clip(pt_ii_next[:,1], 0, binary.shape[1]-1)
        
        if track_flow:
            tracks.append(pt_ii_next)
        else:
            tracks[-1] = pt_ii_next.copy() # copy over. 
        
        if debug_viz:
            plt.figure(figsize=(10,10))
            plt.imshow(binary)
            plt.plot(pt_ii_next[:,1], pt_ii_next[:,0], 'r.')
            plt.show()
        
    tracks = np.array(tracks)
    
    cell_seg_connected_original, cell_seg_connected, votes_grid_acc = connected_components_pts_2D( pts=tracks[-1], 
                                                                                                    pts0=pts, 
                                                                                                    shape=binary.shape[:2], 
                                                                                                    smooth_sigma=smooth_sigma, 
                                                                                                    thresh_factor=thresh_factor, 
                                                                                                    mask=mask,
                                                                                                    min_area=min_area)

    return cell_seg_connected_original, cell_seg_connected, tracks, votes_grid_acc



def gradient_watershed3D_binary(binary, 
                                gradient_img=None, 
                                divergence_rescale=True, 
                                smooth_sigma=1, 
                                smooth_gradient=1, 
                                delta=1, 
                                n_iter=100, 
                                min_area=5, 
                                eps=1e-12, 
                                thresh_factor=None, 
                                mask=None,
                                debug_viz=False):
    
    """ parses the instance level segmentation implicitly given as an input binary or a vector field. 
    The algorithm works as an inverse watershed.
    
    Step 1: a grid of points is seeds on the image
    Step 2: points are propagated for n_iter according to the gradient_img, condensing towards cell centers implicitly implied by the gradient image.
    Step 3: individual cluster centers are found by binarisation and connected component, removing objects < min_area
    
    result is an integer image the same size as binary. 

    Parameters
    ----------
    binary : (MxNxL) numpy array
        input binary image defining the voxels that need labeling
  	gradient_img :  (MxNxLx3) numpy array
        This is a gradient field such as that from applying np.array(np.gradient(img)).transpose(1,2,3,0) where img is a potential such as a distance transform or probability map. 
    divergence_rescale : 
        If True, the gradient_img is scaled by the divergence which is equivalent to the mean curvature, this helps to prevent early breakage for tube-like structures.   
    smooth_sigma : scalar
        controls the catchment area for identifying distinct cells at the final propagation position. Smaller smooth_sigma leads to more oversegmentation. 
    smooth_gradient : scalar
    	the isotropic sigma value controlling the Gaussian smoothing of the gradient field. More smoothing results in more cells grouped together
    delta: scalar
    	the voxel size to propagate grid points per iteration. Related to the stability. If too small takes too long. If too large, might not converge. if delta=1, takes a 1 voxel step. 
    n_iter: int 
        the number of iterations to run. (To do: monitor convergence and break early to improve speed)
    min_area: scalar
        volume of cells < min_area are removed. 
    eps: float
        a small number for numerical stability
    thresh_factor: scalar
        The final cells are identified by thresholding on a threshold mean+thresh_factor*std. Thresh_factor controls what is an object prior to connected components analysis 
    mask: (MxNxL) numpy array
        optional binary mask to gate the region to parse labels for.
    debug_viz: bool
        if True, visualise the position of the points at every algorithm iteration. 
        
    Returns
    -------
    cell_seg_connected_original : (MxNxL)
        an integer image where each unique int > 0 relates to a unique object such that object 1 is retrieved by cell_seg_connected_original==1.
        
    """
    
    import scipy.ndimage as ndimage
    import numpy as np 
    import skimage.morphology as skmorph
    import pylab as plt 
    import skimage.measure as skmeasure 
    import skimage.segmentation as sksegmentation 
    from tqdm import tqdm 
    from .plotting import set_axes_equal
    
    if gradient_img is not None:
        sdf_normals = gradient_img.transpose(3,0,1,2) # use the supplied gradients! 
        sdf_normals = sdf_normals * binary[None,...]
    else:
        # compute the signed distance transform
        sdf_normals, sdf_binary = surf_normal_sdf(binary, return_sdf=True, smooth_gradient=smooth_gradient, eps=eps, norm_vectors=True)
        sdf_normals = sdf_normals * binary[None,...]
    
    if divergence_rescale:
        # rescale the speed
        curvature_3D = mean_curvature_sdf(sdf_normals)
        curvature_3D = _normalize99(curvature_3D, lower=0.01,upper=99) # rescales to a factor between 0-1
        sdf_normals = sdf_normals * curvature_3D[None,...] # multiplicative factor rescaling 
    
    grid =  np.zeros(binary.shape, dtype=np.int32)
    pts = np.argwhere(binary>0) # (N,ndim)
    
    tracks = [pts]
    
    for ii in tqdm(np.arange(n_iter)):
        pt_ii = tracks[-1].copy()
        
        """
        interp helps!. 
        """
        pts_vect_ii = np.array([_interp3(pt_ii, binary.shape, I_ref=sdf_normals[ch], method='linear', cast_uint8=False) for ch in np.arange(len(sdf_normals))]).T
        # pts_vect_ii = sdf_normals[:,
        #                           pt_ii[...,0].astype(np.int32), 
        #                           pt_ii[...,1].astype(np.int32), 
        #                           pt_ii[...,2].astype(np.int32)].T  # direct lookup - not interp!. 
        pts_vect_ii = pts_vect_ii / (np.linalg.norm(pts_vect_ii, axis=-1)[:,None] + 1e-20)
        
        pt_ii_next = pt_ii + delta*pts_vect_ii
            
        pt_ii_next[:,0] = np.clip(pt_ii_next[:,0], 0, binary.shape[0]-1)
        pt_ii_next[:,1] = np.clip(pt_ii_next[:,1], 0, binary.shape[1]-1)
        pt_ii_next[:,2] = np.clip(pt_ii_next[:,2], 0, binary.shape[2]-1)
        
        tracks[-1] = pt_ii_next # overwrite 
        
        # plt.figure(figsize=(10,10))
        # plt.imshow(binary.max(axis=0))
        # plt.plot(pt_ii_next[:,2], 
        #          pt_ii_next[:,1], 'r.')
        # plt.show()
        
        if debug_viz:
            sampling = 100
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_proj_type('ortho') # this works better!.
            ax.set_box_aspect(aspect = (1,1,1)) # this works. 
            # ax.scatter(v_watertight[::sampling,0], 
            #             v_watertight[::sampling,1], 
            #             v_watertight[::sampling,2], 
            #             c='k', s=1, alpha=0.0)#all_labels_branches[np.squeeze(all_dists)<20], s=1)
            ax.scatter(pt_ii_next[::sampling,0], 
                       pt_ii_next[::sampling,1],
                       pt_ii_next[::sampling,2], c='r',s=1)
            # ax.scatter(centroids3D_from_xz[:,0], 
            #            centroids3D_from_xz[:,1],
            #            centroids3D_from_xz[:,2], c='g',s=10)
            # ax.scatter(centroids3D_from_yz[:,0], 
            #            centroids3D_from_yz[:,1],
            #            centroids3D_from_yz[:,2], c='b',s=10)
            # # ax.scatter(skel3D_coords[:,0], 
            # #             skel3D_coords[:,1],
            # #             skel3D_coords[:,2], c='k',s=5, alpha=1)
            # ax.view_init(-90,0)
            # ax.view_init(0,180)
            ax.view_init(180,0)
            # ax.set_xlim([0,binary.shape[0]]) # why is this plot not good? 
            # ax.set_ylim([0,binary.shape[1]])
            # ax.set_zlim([0,binary.shape[2]])
            set_axes_equal(ax)
            plt.show()
        
    tracks = np.array(tracks)
    

    # parse ... 
    votes_grid_acc = np.zeros(binary.shape)
    votes_grid_acc[(tracks[-1][:,0]).astype(np.int), 
                   (tracks[-1][:,1]).astype(np.int),
                   (tracks[-1][:,2]).astype(np.int)] += 1. # add a vote. 
                   
    # smooth to get a density (fast KDE estimation)
    votes_grid_acc = ndimage.gaussian_filter(votes_grid_acc, sigma=smooth_sigma)  
    
    if thresh_factor is not None:
        if mask is not None:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc[mask]) + thresh_factor*np.std(votes_grid_acc[mask])
        else:
            votes_grid_binary = votes_grid_acc >np.mean(votes_grid_acc) + thresh_factor*np.std(votes_grid_acc)
    else:
        votes_grid_binary = votes_grid_acc > np.mean(votes_grid_acc) # just threshold over the mean. 
        
    cell_seg_connected = skmeasure.label(votes_grid_binary, connectivity=2)
    cell_uniq_regions = np.setdiff1d(np.unique(cell_seg_connected),0)
    if len(cell_uniq_regions)>0:
        props = skmeasure.regionprops(cell_seg_connected)
        areas = np.hstack([re.area for re in props])
        invalid_areas = cell_uniq_regions[areas<=min_area]
    
        for invalid in invalid_areas:
            cell_seg_connected[cell_seg_connected==invalid] = 0
        
    if cell_seg_connected.max() > 0:
        cell_seg_connected = sksegmentation.relabel_sequential(cell_seg_connected)[0]
    
    
    cell_seg_connected_original = np.zeros_like(cell_seg_connected)
    cell_seg_connected_original[(pts[:,0]).astype(np.int), 
                                (pts[:,1]).astype(np.int),
                                (pts[:,2]).astype(np.int)] = cell_seg_connected[(tracks[-1][:,0]).astype(np.int), 
                                                                                (tracks[-1][:,1]).astype(np.int),
                                                                                (tracks[-1][:,2]).astype(np.int)]
                                              
    # if mask is not None:
    #     cell_seg_connected[mask == 0] = 0
        
    # plt.figure(figsize=(10,10))
    # plt.imshow(cell_seg_connected.max(axis=0))
    # plt.show()
    
    # plt.figure(figsize=(10,10))
    # plt.imshow(cell_seg_connected_original.max(axis=0))
    # plt.show()
    
    # return cell_seg_connected_original    

    # ah... didn't make it here? 
    cell_seg_connected_original, cell_seg_connected, votes_grid_acc = connected_components_pts_3D( pts=tracks[-1], 
                                                                                                    pts0=pts, 
                                                                                                    shape=binary.shape[:3], 
                                                                                                    smooth_sigma=smooth_sigma, 
                                                                                                    thresh_factor=thresh_factor, 
                                                                                                    mask=mask,
                                                                                                    min_area=min_area)
    
    
    # plt.figure(figsize=(10,10))
    # plt.imshow(cell_seg_connected.max(axis=0))
    # plt.show()
    
    # plt.figure(figsize=(10,10))
    # plt.imshow(cell_seg_connected_original.max(axis=0))
    # plt.show()
    
    return cell_seg_connected_original, cell_seg_connected, tracks, votes_grid_acc













