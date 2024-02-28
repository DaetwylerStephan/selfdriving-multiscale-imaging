# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 03:04:09 2023

@author: fyz11
"""

import numpy as np 

def metrics_np(y_true, y_pred, metric_name, metric_type='standard', drop_last = True, mean_per_class=False, verbose=False):
    """ 
    Compute mean metrics of two segmentation masks, via numpy.
    
    IoU(A,B) = |A & B| / (| A U B|)
    Dice(A,B) = 2*|A & B| / (|A| + |B|)
    
    Args:
        y_true: true masks, one-hot encoded.
        y_pred: predicted masks, either softmax outputs, or one-hot encoded.
        metric_name: metric to be computed, either 'iou' or 'dice'.
        metric_type: one of 'standard' (default), 'soft', 'naive'.
          In the standard version, y_pred is one-hot encoded and the mean
          is taken only over classes that are present (in y_true or y_pred).
          The 'soft' version of the metrics are computed without one-hot 
          encoding y_pred.
          The 'naive' version return mean metrics where absent classes contribute
          to the class mean as 1.0 (instead of being dropped from the mean).
        drop_last = True: boolean flag to drop last class (usually reserved
          for background class in semantic segmentation)
        mean_per_class = False: return mean along batch axis for each class.
        verbose = False: print intermediate results such as intersection, union
          (as number of pixels).
    Returns:
        IoU/Dice of y_true and y_pred, as a float, unless mean_per_class == True
          in which case it returns the per-class metric, averaged over the batch.
    
    Inputs are B*W*H*N tensors, with
        B = batch size,
        W = width,
        H = height,
        N = number of classes
    """
    
    assert y_true.shape == y_pred.shape, 'Input masks should be same shape, instead are {}, {}'.format(y_true.shape, y_pred.shape)
    assert len(y_pred.shape) == 4, 'Inputs should be B*W*H*N tensors, instead have shape {}'.format(y_pred.shape)
    
    flag_soft = (metric_type == 'soft')
    flag_naive_mean = (metric_type == 'naive')
    
    num_classes = y_pred.shape[-1]
    # if only 1 class, there is no background class and it should never be dropped
    drop_last = drop_last and num_classes>1
    
    if not flag_soft:
        if num_classes>1:
            # get one-hot encoded masks from y_pred (true masks should already be in correct format, do it anyway)
            y_pred = np.array([ np.argmax(y_pred, axis=-1)==i for i in range(num_classes) ]).transpose(1,2,3,0)
            y_true = np.array([ np.argmax(y_true, axis=-1)==i for i in range(num_classes) ]).transpose(1,2,3,0)
        else:
            y_pred = (y_pred > 0).astype(int)
            y_true = (y_true > 0).astype(int)
    
    # intersection and union shapes are batch_size * n_classes (values = area in pixels)
    axes = (1,2) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) # or, np.logical_and(y_pred, y_true) for one-hot
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection # or, np.logical_or(y_pred, y_true) for one-hot
    
    if verbose:
        print('intersection (pred*true), intersection (pred&true), union (pred+true-inters), union (pred|true)')
        print(intersection, np.sum(np.logical_and(y_pred, y_true), axis=axes), union, np.sum(np.logical_or(y_pred, y_true), axis=axes))
    
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    
    metric = {'iou': iou, 'dice': dice}[metric_name]
    
    # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
    mask =  np.not_equal(union, 0).astype(int)
    # mask = 1 - np.equal(union, 0).astype(int) # True = 1
    
    if drop_last:
        metric = metric[:,:-1]
        mask = mask[:,:-1]
    
    # return mean metrics: remaining axes are (batch, classes)
    # if mean_per_class, average over batch axis only
    # if flag_naive_mean, average over absent classes too
    if mean_per_class:
        if flag_naive_mean:
            return np.mean(metric, axis=0)
        else:
            # mean only over non-absent classes in batch (still return 1 if class absent for whole batch)
            return (np.sum(metric * mask, axis=0) + smooth)/(np.sum(mask, axis=0) + smooth)
    else:
        if flag_naive_mean:
            return np.mean(metric)
        else:
            # mean only over non-absent classes
            class_count = np.sum(mask, axis=0)
            return np.mean(np.sum(metric * mask, axis=0)[class_count!=0]/(class_count[class_count!=0]))
        
def mean_iou_np(y_true, y_pred, **kwargs):
    """
    Compute mean Intersection over Union of two segmentation masks, via numpy.
    
    Calls metrics_np(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """
    return metrics_np(y_true, y_pred, metric_name='iou', **kwargs)

def mean_dice_np(y_true, y_pred, **kwargs):
    """
    Compute mean Dice coefficient of two segmentation masks, via numpy.
    
    Calls metrics_np(y_true, y_pred, metric_name='dice'), see there for allowed kwargs.
    """
    return metrics_np(y_true, y_pred, metric_name='dice', **kwargs)


"""
functions for 2D cell comparison
"""
def _match_cells(labels1, labels2, com1, com2, K=10, bg_label=0):
    
    """
    labels1 - ground truth 
    labels2 - predicted labels
    com1 - center-of-mass for labels1
    com2 - center-of-mass for labels2
    K - # of nearest neighbor candidates. 
    """
    from scipy.optimize import linear_sum_assignment # brute force is the absolute gold-standard, but can we do this really fast? and without needing to do 
    from sklearn.neighbors import NearestNeighbors
    
    uniq1 = np.setdiff1d(np.unique(labels1), bg_label)
    uniq2 = np.setdiff1d(np.unique(labels2), bg_label)
    
    n1 = len(uniq1)
    n2 = len(uniq2)
    
    # initialise matrix. 
    sim_matrix = np.zeros((n1,n2))
    dice_matrix = np.zeros((n1,n2))
    
    # turn into numpy array. 
    com1 = np.vstack(com1)
    com2 = np.vstack(com2)
    
    # nearest neighbor match on centroids, then bipartite matching on iou !
    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(com1)
    _, indices = nbrs.kneighbors(com2)
#        print(indices.shape)
#        print(len(com1))
    
    for j in range(len(com2)):
        cand_i = indices[j]
 
        if len(cand_i) > 0:
            for i in range(len(cand_i)):
                mask1 = labels1 == uniq1[cand_i[i]]
                mask2 = labels2 == uniq2[j]
                intersection = np.sum(np.abs(mask1*mask2))
                union = np.sum(mask1) + np.sum(mask2) - intersection
                
                # jaccard. 
                overlap = intersection / float(union + 1e-8)
                
                # dice?
                dice = 2*intersection / float(np.sum(mask1) + np.sum(mask2) + 1e-8)
                
#                    print(overlap)
                sim_matrix[cand_i[i],j] = np.clip(overlap, 0, 1)
                dice_matrix[cand_i[i],j] = np.clip(dice, 0, 1)
    
    # hungarian.         
    ind_i, ind_j = linear_sum_assignment(1-sim_matrix) # need to reverse this (distance)
    
    iou_pair = sim_matrix[ind_i, ind_j].copy()
    dice_pair = dice_matrix[ind_i, ind_j].copy()
    
    valid = iou_pair>0 # must have non-zero overlap!
    
    ind_i = ind_i[valid>0].copy()
    ind_j = ind_j[valid>0].copy()
    iou_pair = iou_pair[valid>0].copy()
    dice_pair = dice_pair[valid>0].copy()
    
    # return at the end also the sim matrix. 
    return ind_i, ind_j, iou_pair, dice_pair, sim_matrix
    

"""
move to segmentation?
"""
def remove_small_labelled_objects(labels, minsize=64):
    
    import skimage.measure as skmeasure
    import numpy as np 
    
    unique_label_ = np.setdiff1d(np.unique(labels), 0)
    
    regprops = skmeasure.regionprops(labels)
    regareas = np.hstack([re.area for re in regprops])
    
    labels_out = labels.copy()
    
    remove_reg = unique_label_[regareas<=minsize]
    
    if len(remove_reg)>0:
        for rr in remove_reg:
            labels_out[labels==rr] = 0 # set to bg
            
    return labels_out
        

def compute_metrics_cells(labels_true, 
                          labels_pred, 
                          bg_label=0, 
                          K=15, 
                          iou_thresh=0, 
                          eps=1e-5,
                          debug_viz=False):
    """
    processes a list of images.
    
    """
    from skimage.measure import label
    from skimage.filters import threshold_otsu
    # import scipy.ndimage.measurements as scipy_measure
    import scipy.ndimage as ndimage 
    import pylab as plt 
    
    n_images = len(labels_true)
    # based on the overlap we assign and compute the AP based on the segmentations.
#    if thresh is not None:
    stats = [] # n_GT, n_Pred, n_match, overlap_score. 
    match_props = []
    
    for ii in range(n_images):
        label_true = labels_true[ii].copy()
        label_pred = labels_pred[ii].copy()
        
        unique_label_true = np.setdiff1d(np.unique(label_true), bg_label)
        unique_label_pred = np.setdiff1d(np.unique(label_pred), bg_label)
        
        if debug_viz:
            plt.figure()
            plt.imshow(label_pred)
            plt.figure()
            plt.imshow(label_true)
            plt.show()
            
        """
        Use a nearest neighbour type matching to expedite the instance matching. 
        """
        com_true = ndimage.center_of_mass(label_true>0, labels=label_true, index=unique_label_true); com_true = np.vstack(com_true)
        com_pred = ndimage.center_of_mass(label_pred>0, labels=label_pred, index=unique_label_pred); com_pred = np.vstack(com_pred)
    
    
        # Solve the matching problem based on iou (using knn as a prefilter)
        gt_i, pred_j, iou_ij, dice_ij, iou_matrix = _match_cells(label_true, 
                                                                 label_pred, 
                                                                 com_true, 
                                                                 com_pred, 
                                                                 K=np.minimum(K, len(label_pred)), 
                                                                 bg_label=bg_label)
        
        # gate if only restricting to matches above a given threshold!. 
        val_index = iou_ij > iou_thresh 
        
        gt_i = gt_i[val_index]
        pred_j = pred_j[val_index] 
        iou_ij = iou_ij[val_index]
        dice_ij = dice_ij[val_index]
        
        match_dict = {'gt_index': gt_i ,
                      'pred_index': pred_j, 
                      'iou_gt_pred': iou_ij, 
                      'dice_gt_pred': dice_ij, 
                      'iou_matrix': iou_matrix,
                      'gt_com': com_true,
                      'pred_com': com_pred, 
                      'matched_labels_gt_pred': [unique_label_true[gt_i], unique_label_pred[pred_j]]}
        
        """
        Compute the stats of matching 
        """
        n_match = len(pred_j)
        n_GT = len(unique_label_true)
        n_Pred = len(unique_label_pred)
        
        pre = n_match/float(n_Pred + eps)
        rec = n_match/float(n_GT + eps)
        f1 = 2*pre*rec / (pre + rec)
        
        iou = np.mean(iou_ij) # mean_iou matrix. 
        dice = np.mean(dice_ij)
        
        stats.append([n_GT, n_Pred, n_match, pre, rec, f1, iou, dice])
        match_props.append(match_dict)
        
        
    return np.vstack(stats), match_props
