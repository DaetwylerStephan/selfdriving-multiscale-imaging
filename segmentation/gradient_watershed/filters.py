# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 21:37:54 2023

@author: fyz11

helper filters. 
"""

import numpy as np 

def imadjust(vol, p1, p2): 
    import numpy as np 
    from skimage.exposure import rescale_intensity
    # this is based on contrast stretching and is used by many of the biological image processing algorithms.
    p1_, p2_ = np.percentile(vol, (p1,p2))
    vol_rescale = rescale_intensity(vol, in_range=(p1_,p2_))
    return vol_rescale

def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)
    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )
    if clip:
        x = np.clip(x,0,1)
    return x

def normalize(x, pmin=2, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0, option=1,ploton=False):
    
    """
    Anisotropic diffusion in 2D 

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """
    
    # import skimage.filters as flt
    import scipy.ndimage.filters as flt
    import warnings 
    import skimage.io as io
    import matplotlib
    import numpy as np 
    
    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep

        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

        ax1.imshow(img,interpolation='nearest')
        ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in np.arange(1,niter):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        if 0<sigma:
            deltaSf=flt.gaussian_filter(deltaS,sigma);
            deltaEf=flt.gaussian_filter(deltaE,sigma);
        else: 
            deltaSf=deltaS;
            deltaEf=deltaE;
            
        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaSf/kappa)**2.)/step[0]
            gE = np.exp(-(deltaEf/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaSf/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaEf/kappa)**2.)/step[1]

        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        # update the image
        imgout += gamma*(NS+EW)

        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return imgout


def relabel_slices(labelled, bg_label=0):
    
    import numpy as np 
    max_ID = 0 
    labelled_ = []
    for lab in labelled:
        lab[lab>bg_label] = lab[lab>bg_label] + max_ID # only update the foreground! 
        labelled_.append(lab)
        max_ID = np.max(lab)+1
        
    labelled_ = np.array(labelled_)
    
    return labelled_


def filter_segmentations_axis(labels, window=3, min_count=1):
    """ according to the persistence along 1 direction. - this works and is relatively fast. 
    """
    import skimage.measure as skmeasure 
    
    labels_filt = []
    N = len(labels)
    offset = window//2
    
    labels_pad = np.pad(labels, [[offset,offset], [0,0], [0,0]], mode='reflect')
    
    for ss_ii in np.arange(N):
        # base
        ss = labels_pad[ss_ii+offset].copy()
        ss_copy = ss.copy()
        
        # iterate over the unique regions. 
        uniq_reg = np.setdiff1d(np.unique(ss),0)
        
        for reg in uniq_reg:
            mask = ss == reg
            
            checks = np.zeros(2*offset)
            cnt = 0
            for ii in np.arange(0, 2*offset+1):
                # now we are going to apply. 
                if ii != offset:
                    # print(ii)
                    mask_ii = labels_pad[ss_ii+cnt, mask>0].copy()
                    
                    # print(np.sum(mask_ii>0))
                    if np.sum(mask_ii>0) >= min_count:
                        checks[cnt] += 1 # append true. 
                    
                    cnt+=1
                # else:
                #     checks[cnt] += 1
                #     cnt+=1
                    
                    
            # print(checks)
            # if np.max(checks) < offset: # less than half i.e. not persistent # so this is wrong!. --- what i was going to do is the below. 
            # i think we do need to check continguousness!. 
            
            # check contiguity! 
            contigs = skmeasure.label(checks)
            
            # print(contigs)
            
            if contigs.max() > 0:
                uniq_contig = np.setdiff1d(np.unique(contigs), 0)
                lengths = [np.sum(contigs==cc) for cc in uniq_contig]
                mask = contigs == uniq_contig[np.argmax(lengths)]
                
                # print(mask)
                # print(contigs)
                # print(np.sum(mask)>= window //2 + 1)
                # print('---')
                
                # check the length of mask
                if np.sum(mask)>= window //2 + 1:
                    # if sufficiently long. 
                    if mask[offset-1] == 0 and mask[offset] == 0: # that is doesn't cover either of this !. 
                        ss_copy[ss==reg] = 0
                else:
                    # missing clause! 
                    ss_copy[ss==reg] = 0
            else:
                ss_copy[ss==reg] = 0 # zero it
            # if np.sum(checks) < len(checks): 
            #     ss_copy[ss==reg] = 0 # zero it
                
        labels_filt.append(ss_copy)
        
    labels_filt = np.array(labels_filt)
    
    return labels_filt



def filter_segmentations_axis_IoU(labels, window=3, iou_var = 0.1):
    """ according to the persistence along 1 direction of the IoU with respect to the middle reference shape.  
    """
    
    labels_filt = []
    N = len(labels)
    offset = window//2
    
    labels_pad = np.pad(labels, [[offset,offset], [0,0], [0,0]], mode='reflect')
    
    for ss_ii in np.arange(N):
        # base
        ss = labels_pad[ss_ii+offset].copy()
        ss_copy = ss.copy()
        
        # iterate over the unique regions. 
        uniq_reg = np.setdiff1d(np.unique(ss),0)
        
        for reg in uniq_reg:
            mask = ss == reg
            
            iou_checks = np.zeros(2*offset)
            cnt = 0
            for ii in np.arange(0, 2*offset+1):
                # now we are going to apply. 
                if ii != offset:
                    # print(ii)
                    mask_ii = labels_pad[ss_ii+cnt, mask>0].copy()
                    
                    if mask_ii.max()>0:
                        uniq_regions_mask_ii = np.setdiff1d(np.unique(mask_ii),0)
                        largest = uniq_regions_mask_ii[np.argmax([np.sum(labels_pad[ss_ii+cnt]==rr) for rr in uniq_regions_mask_ii])]
                        largest_mask = labels_pad[ss_ii+cnt]==largest
                        # compute the IoU overlap. 
                        overlap = np.sum(np.logical_and(largest_mask, mask))
                        iou = np.sum(np.logical_and(largest_mask, mask)) / (np.sum(mask)+np.sum(largest_mask) - overlap)
                        iou_checks[cnt] = iou
                    else:
                        iou_checks[cnt] = 0 
                    
                    # # print(np.sum(mask_ii>0)>0)
                    # if np.sum(mask_ii>0) >= min_count:
                    #     checks[cnt] += 1 # append true. 
                    cnt+=1
                # else:
                    
            # print(checks)
            # if np.max(checks) < offset: # less than half i.e. not persistent # so this is wrong!. --- what i was going to do is the below. 
            # print(iou_checks, np.std(iou_checks))
            if np.std(iou_checks) > iou_var: 
                ss_copy[ss==reg] = 0 # zero it
                
        labels_filt.append(ss_copy)
        
    labels_filt = np.array(labels_filt)
    
    return labels_filt



# potentially extend this to handle anistropic! 
def smooth_vol(vol_binary, ds=4, smooth=5):
    
    from skimage.filters import gaussian
    from scipy.ndimage import gaussian_filter
    import skimage.transform as sktform
    import numpy as np 
    
    small = sktform.resize(vol_binary, np.array(vol_binary.shape)//ds, preserve_range=True)
    small = gaussian_filter(small, sigma=smooth)
    
    return sktform.resize(small, np.array(vol_binary.shape), preserve_range=True)


def smooth_vol_anisotropic(vol_binary, ds=4, smooth=[5,5,5]):
    
    from skimage.filters import gaussian
    from scipy.ndimage import gaussian_filter, gaussian_filter1d
    import skimage.transform as sktform
    import numpy as np 
    
    small = sktform.resize(vol_binary, np.array(vol_binary.shape)//ds, preserve_range=True)
    # small = gaussian_filter(small, sigma=smooth)
    for axis in np.arange(len(smooth)):
        small = gaussian_filter1d(small, sigma=smooth[axis], axis=axis) # apply separable filtering 
    
    return sktform.resize(small, np.array(vol_binary.shape), preserve_range=True)


# =============================================================================
# for cleaning labels
# =============================================================================
def remove_small_labels(labels, min_size=64):
    
    import skimage.measure as skmeasure
    
    if labels.max()>0:
        uniq_regions = np.setdiff1d(np.unique(labels),0)
        props = skmeasure.regionprops(labels)
    
        areas = np.hstack([re.area for re in props])
        # print(areas)
        regions_remove = uniq_regions[areas<min_size]
        
        labels_new = labels.copy()
        for reg in regions_remove:
            labels_new[labels==reg] = 0 # set to background
        
        return labels_new
    else:
        
        return labels
    
    
def largest_component_vol(vol_binary, connectivity=1):
    
    from skimage.measure import label, regionprops
    import numpy as np 
    
    vol_binary_labelled = label(vol_binary, connectivity=connectivity)
    # largest component.
    vol_binary_props = regionprops(vol_binary_labelled)
    vol_binary_vols = [re.area for re in vol_binary_props]
    vol_binary = vol_binary_labelled == (np.unique(vol_binary_labelled)[1:][np.argmax(vol_binary_vols)])
    
    return vol_binary
    

def remove_small_obj_and_keep_largest(labels, min_size=64, connectivity=2):
    
    import skimage.measure as skmeasure
    
    if labels.max()>0:
        uniq_regions = np.setdiff1d(np.unique(labels),0)
        props = skmeasure.regionprops(labels)
    
        areas = np.hstack([re.area for re in props])
        
        labels_new = np.zeros_like(labels)
        
        for re_ii in np.arange(len(uniq_regions)):
            
            area_ii = areas[re_ii]
            if area_ii < min_size:
                continue
            else:
                mask = labels==uniq_regions[re_ii]
                mask = largest_component_vol(mask, connectivity=connectivity)
                labels_new[mask>0] = re_ii
        
        return labels_new
    else:
        
        return labels
    
    
def expand_masks(label_seeds, binary, dist_tform=None):
    
    import skimage.measure as skmeasure 
    from skimage.segmentation import watershed
    
    if dist_tform is None:
        
        from .flows import distance_transform_labels_fast 
        dist_tform = distance_transform_labels_fast(skmeasure.label(binary, connectivity=2))
    
    # use the initial labels as seed
    seeds = label_seeds * binary
    
    labels_refine = watershed(-dist_tform, seeds, mask=binary) # this fills in the binary as much as possible. 
    
    # the remainder we can label otherwise. 
    remainder = np.logical_and(binary>0, labels_refine==0)
    remainder_label = skmeasure.label(remainder)
    remainder_label[remainder>0] = remainder_label[remainder>0] + label_seeds.max()
    
    
    return labels_refine


def expand_masks2D(label_seeds, binary, dist_tform=None):
    
    import skimage.measure as skmeasure 
    from skimage.segmentation import watershed
    
    if dist_tform is None:
        
        from .flows import distance_transform_labels_fast 
        dist_tform = distance_transform_labels_fast(skmeasure.label(binary, connectivity=2)) # this is running 2D slice by slice!. 
    
    # use the initial labels as seed
    seeds = label_seeds * binary # 
    
    labels_refine = []
    
    for dd in np.arange(len(binary)):
        labels_refine_slice = watershed(-dist_tform[dd], seeds[dd], mask=binary[dd]) # this fills in the binary as much as possible. 
    
        # the remainder we can label otherwise. 
        remainder = np.logical_and(binary[dd]>0, labels_refine_slice==0)
        remainder_label = skmeasure.label(remainder)
        remainder_label[remainder>0] = remainder_label[remainder>0] + label_seeds.max()
        
        labels_refine_slice[remainder>0] = remainder_label[remainder>0].copy()
        labels_refine.append(labels_refine_slice)
    labels_refine = np.array(labels_refine)
    
    return labels_refine
    

def remove_eccentric_shapes(labels, min_size=20, stretch_cutoff=15):

    ##### get the objects 
    import numpy as np 
    
    labels_filter = labels.copy()

    obj_area = []
    obj_ecc = [] 
    
    uniq_labs_3D = np.setdiff1d(np.unique(labels), 0)
    
    for lab in uniq_labs_3D[:]:
        mask = labels==lab 
        obj_area.append(np.nansum(mask))
        
        pts_mask = np.vstack(np.argwhere(mask>0))
        
        if len(pts_mask) > min_size:
            cov = np.cov((pts_mask-pts_mask.mean(axis=0)[None,:]).T)
            eigs = np.linalg.eigvalsh(cov)
            
            stretch_ratio = np.max(np.max(np.abs(eigs))/np.abs(eigs))
            
            if stretch_ratio > stretch_cutoff :
                labels_filter[mask] = 0
            
            obj_ecc.append(stretch_ratio)
        else:
            obj_ecc.append(np.nan) # no obj_ecc
            labels_filter[mask] = 0
            
    return labels_filter, (obj_area, obj_ecc)


# =============================================================================
# for fusion
# =============================================================================
def entropy_mean(arrays, ksize=3,alpha = 0.5, eps=1e-20):
        
    import skimage.filters.rank as skfilters_rank #import entropy 
    import skimage.morphology as skmorph
    
    ents = np.array([1./(skfilters_rank.entropy(array, selem=skmorph.ball(ksize)) + alpha) for array in arrays])
    
    v = np.sum(ents*arrays, axis=0) / (ents.sum(axis=0)+eps)
    
    return v


def var_filter(img, ksize):
    import scipy.ndimage as ndimage
    import numpy as np 
    
    win_mean = ndimage.uniform_filter(img,ksize)
    win_sqr_mean = ndimage.uniform_filter(img**2, ksize)
    win_var = win_sqr_mean - win_mean**2
    
    return win_var


def var_combine_registration(arrays, ksize=3, alpha=1., eps=1e-20):
    """
    :param arrays:
    :param ksize: sets neighborhood to look at variance
    :param alpha:
    :param eps:
    :return:
    """
    import skimage.filters.rank as skfilters_rank  # import entropy
    import skimage.morphology as skmorph

    weights = np.array([(var_filter(array, ksize=ksize) + alpha) for array in arrays])

    v = np.sum(weights * arrays, axis=0) / (weights.sum(axis=0) + eps)

    return v

def var_combine(arrays, ksize=3, alpha = 1., eps=1e-20):
    """

    :param arrays:
    :param ksize: sets neighborhood to look at variance
    :param alpha:
    :param eps:
    :return:
    """
    import skimage.filters.rank as skfilters_rank #import entropy 
    import skimage.morphology as skmorph
    
    weights = np.array([1./(var_filter(array, ksize=ksize) + alpha) for array in arrays])
    
    v = np.sum(weights*arrays, axis=0) / (weights.sum(axis=0)+eps)
    
    return v


def _distance_to_heat_affinity_matrix(Dmatrix, gamma=None):
    r""" Convert any distance matrix to an affinity matrix by applying a heat kernel.

    .. math:: 
        A = \exp^{\left(\frac{-D^2}{2\sigma^2}\right)}

    where :math:`sigma` is set as the mean distance of :math:`D` or :math:`\gamma` if provided.
    
    Parameters
    ----------
    Dmatrix : (N,N) sparse array
        a scipy.sparse input distance matrix
    gamma : scalar
        the normalisation scale factor of distances

    Returns 
    -------
    A : (N,N) sparse array
        a scipy.sparse output affinity distance matrix

    """
    import numpy as np 
    # import igl
    import scipy.sparse as spsparse

    l = Dmatrix.shape[0]
    A = Dmatrix.copy()
    if gamma is None:
        sigma_D = np.mean(A.data)
    else:
        sigma_D = gamma
    den_D = 2 * (sigma_D ** 2)
    np.exp( -A.data**2/den_D, out=A.data )
    A = A + spsparse.diags(np.ones(l), 0)  # diagonal is 1 by definition. 

    return A 


def diffuse_labels3D(labels_in, guide, clamp=0.99, n_iter=10):
    
    from sklearn.feature_extraction.image import img_to_graph
    from tqdm import tqdm 
    
    graph = img_to_graph(guide) # use gradients
    affinity = _distance_to_heat_affinity_matrix(graph, gamma=None)
    # normalize this.... 
    
    n_labels = np.max(labels_in)+1 # include background!. 
    
    labels = np.zeros((np.prod(labels_in.shape[:3]), n_labels), 
                          dtype=np.float32)
    labels[np.arange(len(labels_in.ravel())), labels_in.ravel()] = 1 # set all the labels 
    
    # diffuse on this.... with label propagation.
    alpha_prop = clamp
    base_matrix = (1.-alpha_prop)*labels
    init_matrix = np.zeros_like(labels) # let this be the new. 
    
    for ii in tqdm(np.arange(n_iter)):
        init_matrix = affinity.dot(init_matrix) + base_matrix
        
    z = np.nansum(init_matrix, axis=1)
    z[z==0] += 1 # Avoid division by 0
    z = ((init_matrix.T)/z).T
    z_label = np.argmax(z, axis=1)
    z_label = z_label.reshape(labels_in.shape)
    
    return z_label
    
    
    

