########
# In this file, the segmented macrophages are postprocessed by
# 1. removing the cancer signal from the label / segmented image by thresholding the cancer cell image and masking the label image
# 2. by removing small labeled volumes (below volume size, determined by minimum_volumesize
#######

import os
import skimage.io as skio
import scipy.ndimage as ndimage
import skimage.filters as skfilters
import skimage.morphology as morphology
import numpy as np
import skimage.color as skcolor
import seaborn as sns

import skimage.morphology as skmorph

try:
    import Segmentation.gradient_watershed.filters as grad_filters
except:
    import gradient_watershed.filters as grad_filters

if __name__ == "__main__":
    # =============================================================================
    # initializations
    # =============================================================================
    """parameters"""
    lateral_axialratio = 3.42  # for (down-)sampling image to isotropic voxels, e.g. 3.42 for axial spacing of 0.4 um and 0.117 nm lateral
    debug = 0
    regions = ["high_stack_003", "high_stack_002", "high_stack_001"]
    thresholds = [112, 120, 125, 130, 135]

    #set_threshold = 135
    set_threshold=125
    minimum_volumesize = 7000

    """init Image-preprocessing class"""
    try:
        from Segmentation.preprocessing_class import ImagePreprocessing as im_preprocess
    except:
        from preprocessing_class import ImagePreprocessing as im_preprocess
    im_preprocessing = im_preprocess()

    # =============================================================================
    # Load the images and generate save folder
    # =============================================================================
    for threshold_iter in thresholds:
        for region in regions:
            experimentfolder_parent = "/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS/"
            experimentfolder_segmentation = os.path.join(experimentfolder_parent, "Experiment0001_highresSeg_connectedComp_multiOtsu/result_segmentation/", region)
            experimentfolder_rawdata =  os.path.join(experimentfolder_parent, "Experiment0001")
            experimentfolder_result = experimentfolder_parent + "Experiment0001_highresSeg_connectedComp_multiOtsu"
            im_preprocessing.mkdir(experimentfolder_result)


            # get all timepoints of folder
            dir_list = os.listdir(experimentfolder_segmentation)
            timepointlist = []
            for path in dir_list:
                if path.startswith('t'):
                    timepointlist.append(path)
            timepointlist.sort()
            print(timepointlist)

            for i_time in timepointlist:
                # =============================================================================
                # define filepaths, load images and make folder
                # =============================================================================

                #labelimage_filepath = os.path.join(experimentfolder_segmentation,i_time,"labels_xy-integrated_gradients-correct_noexpand.tif")
                labelimage_filepath = os.path.join(experimentfolder_segmentation, i_time,
                                                   "labels_xy-connectedcomponents.tif")

                cancerimage_filepath = os.path.join(experimentfolder_rawdata,i_time,region,"1_CH552_000000.tif")

                im_segmentation = skio.imread(labelimage_filepath)
                im_cancer_raw = skio.imread(cancerimage_filepath)

                savefolder = os.path.join(experimentfolder_result,"processed_segmentation_thres" + str(threshold_iter), region, i_time)
                im_preprocessing.mkdir(savefolder)

                # =============================================================================
                # rescale image
                # =============================================================================
                im_cancer = ndimage.zoom(im_cancer_raw, [1., 1. / lateral_axialratio, 1. / lateral_axialratio], order=1, mode='reflect')


                # binary_thresh = skfilters.threshold_otsu(im_cancer)  # automatic threshold doesn't work at timepoints in the end
                im_binary_cancer = im_cancer > threshold_iter  # +0.1;
                morphology.closing(im_binary_cancer,out=im_binary_cancer)
                skio.imsave(os.path.join(savefolder, "im_binary_cancercell" + i_time + ".tif"), np.uint16(im_binary_cancer))

                im_seg_substracted = im_segmentation * (1-im_binary_cancer)
                morphology.closing(im_seg_substracted,out=im_seg_substracted)

                im_seg_substracted_removedSmall = grad_filters.remove_small_labels(im_seg_substracted, min_size=minimum_volumesize)  # this removed ?

                skio.imsave(os.path.join(savefolder, 'labels_xy-connectedcomponents.tif'), np.uint16(im_seg_substracted_removedSmall))

                labels_3D_substracted_color = np.uint8(255 * skcolor.label2rgb(im_seg_substracted_removedSmall.copy(),
                                                                             colors=sns.color_palette('hls', n_colors=16),
                                                                             bg_label=0))

                skio.imsave(os.path.join(savefolder, 'labels_xy-connectedcomponentsRGB.tif'),np.uint8(labels_3D_substracted_color))



