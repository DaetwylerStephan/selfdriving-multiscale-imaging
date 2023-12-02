
import numpy as np
import skimage.io as skio
import skimage.exposure as skexposure
import scipy.ndimage as ndimage
import pylab as plt
import os
import seaborn as sns

from tiler import Tiler, Merger
from tqdm import tqdm
import scipy.io as spio
import skimage.measure as skmeasure
import skimage.morphology as skmorph
import skimage.restoration as skrestoration
import skimage.color as skcolor
import skimage.segmentation as sksegmentation
from pystackreg import StackReg
from tifffile import imread, imwrite

def prepare_segmentations(im, savedeconvfilepath, lateral_axialratio, psffilepath):
    """
    generate preprocessed/deconvolved data

    :param im: image of macrophages
    :param savedeconvfilepath: folder to save the deconvolved file
    :param lateral_axialratio: ratio of lateral to axial spacing
    :param psffilepath: filepath to PSF file for deconvolution
    :return: saved images, ready to segment
    """
    # =============================================================================
    # Resample to isotropic
    # =============================================================================

    im = ndimage.zoom(im, [1., 1. / lateral_axialratio, 1. / lateral_axialratio], order=1, mode='reflect')
    print("resampled images")

    # =============================================================================
    #     Preprocess fully in 3D
    # =============================================================================

    # smooth background
    im_bg = im_preprocessing.smooth_vol(im, ds=5, smooth=5)
    #normalize image
    im = im_preprocessing.normalize(im - im_bg, pmin=2, pmax=99.8, clip=True)

    # deconvolution
    psf_meSPIM = spio.loadmat(psffilepath)['PSF']
    psf_meSPIM = psf_meSPIM / (float(np.sum(psf_meSPIM)))

    im_deconv = skrestoration.wiener(im, psf_meSPIM,
                                     balance=0.5)  # was doing balance=0.5 # use a smaller balance to retain sharper features.
    im_deconv = np.clip(im_deconv, 0, 1)
    im_deconv = np.array(
        [im_preprocessing.anisodiff(ss, niter=15, kappa=1, gamma=0.1, step=(1., 1.), sigma=0, option=1, ploton=False)
         for ss in im_deconv])

    #save image
    skio.imsave(os.path.join(savedeconvfilepath, basename + 'deconv_demix.tif'),
                np.uint16(65535 * im_deconv))

    print("deconvolved data saved")


def generate_binaryimage(im_deconvolved):
    """
    generate binary image from deconvolved image using multi-otsu tresholding
    :param im_deconvolved:
    :return: binary image
    """
    import skimage.morphology as skmorph
    import skimage.filters as skfilters

    # =============================================================================
    # Combine the deconvolved image with variance filters and obtain a binary segmentation mask!.
    # =============================================================================

    dog = im_deconvolved - ndimage.gaussian_filter(im_deconvolved, sigma=1)
    im_ = (im_deconvolved - im_deconvolved.mean()) / (np.std(im_deconvolved));
    im_ = np.clip(im_, 0, 4)
    binary_thresh = skfilters.threshold_multiotsu(im_)
    binary_thresh_upper = skfilters.threshold_otsu(im_)
    im_binary = im_ > (binary_thresh[0]+binary_thresh_upper)/2  # +0.1;

    im_binary = skmorph.remove_small_objects(im_binary, min_size=1000, connectivity=2)

    comb = np.maximum(im_binary * 1, (dog - dog.mean()) / (4 * np.std(dog)))
    im_binary = comb >= 1
    im_binary = skmorph.binary_closing(im_binary, skmorph.ball(1))
    im_binary = ndimage.binary_fill_holes(im_binary)


    skio.imsave(os.path.join(savefolder_file, basename + 'im_binary.tif'),
                np.uint16(im_binary))
    print("binary mask generated")
    return im_binary

def register_stack(imagestack, applymedian=0, reference_flag='mean'):
            """
            Register the stack to account for sample drift that occurs
            :param imagestack: stack to register
            :param applymedian: apply value to zero elements if non zero
            :param reference_flag: how should stackreg register stack
            :return: registered stack
            """

            #calculate median value
            if applymedian==0:
                applymedian = np.median(imagestack[0,:,:])

            print("calculated median: " + str(applymedian))

            # register each frame to the previous (already registered) one
            # this is what the original StackReg ImageJ plugin uses
            sr = StackReg(StackReg.TRANSLATION)
            out_previous = sr.register_transform_stack(imagestack, reference=reference_flag)

            print("stack registered: " + str(out_previous.shape))

            # Assign the median to the zero elements
            out_previous[out_previous == 0] = applymedian

            return out_previous

if __name__ == "__main__":

    # =============================================================================
    # initializations / parameters
    # =============================================================================
    """parameters"""
    lateral_axialratio = 3.42 #for (down-)sampling image to isotropic voxels, e.g. 3.42 for axial spacing of 0.4 um and 0.117 nm lateral
    debug = 0 #flag to debug - run on only one timepoint if set to 1
    region = "high_stack_002"
    alignstackbefore = 0 #do you want to align/register stack before segmentation? (use pystackreg)
    # PSF file path
    psffilepath = "/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/psf/meSPIM_PSF_kernel.mat"

    # experimentfolder = "/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/macrophage_control/20230602_Daetwyler_Xenograft/Experiment0013"
    # experimentfolder_result = experimentfolder + "_highresSeg_raw"

    experimentfolder = "/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS/Experiment0001"
    experimentfolder_result = experimentfolder + "_highresSeg_connectedComp_multiOtsu"

    """init Image-preprocessing class"""
    try:
        from Segmentation.preprocessing_class import ImagePreprocessing as im_preprocess
    except:
        from preprocessing_class import ImagePreprocessing as im_preprocess
    im_preprocessing = im_preprocess()

    # =============================================================================
    # prepare segmentations - check how many timepoints to segment
    # =============================================================================

    # get all timepoints of folder
    dir_list = os.listdir(experimentfolder)
    timepointlist = []
    for path in dir_list:
        if path.startswith('t'):
            timepointlist.append(path)
    timepointlist.sort()
    print(timepointlist)

    # in case of debugging, only run timepoint zero
    if debug == 1:
        timepointlist = ['t00008']
        i_time = 't00008'

    for i_time in timepointlist:

        # =============================================================================
        # Generate filepaths and folders
        # =============================================================================
        print(i_time)

        # construct filepaths
        imagefilepath = os.path.join(experimentfolder, i_time, region,'1_CH488_000000.tif')
        basename = os.path.split(imagefilepath)[-1].split('.tif')[0]
        print(imagefilepath)

        savefolder_deconvolution = os.path.join(experimentfolder_result, "deconvolution", region, i_time)
        savefolder_registered = os.path.join(experimentfolder_result, "registered", region, i_time)
        savefolder_registered_max_xy = os.path.join(experimentfolder_result, "registered_max", region, "max_xy")
        savefolder_registered_max_xz = os.path.join(experimentfolder_result, "registered_max", region, "max_xz")
        savefolder_file = os.path.join(experimentfolder_result, "result_segmentation", region, i_time)

        #make folders
        im_preprocessing.mkdir(savefolder_deconvolution)
        im_preprocessing.mkdir(savefolder_registered)
        im_preprocessing.mkdir(savefolder_registered_max_xy)
        im_preprocessing.mkdir(savefolder_registered_max_xz)
        im_preprocessing.mkdir(savefolder_file)

        # =============================================================================
        # Load image to segment
        # =============================================================================

        im = skio.imread(imagefilepath)
        print("loaded images")

        # =============================================================================
        # 3D register stacks if there was a lot of movement / jitters
        # =============================================================================

        if alignstackbefore==1:
            print("register stack plane by plane")
            im = register_stack(im, reference_flag='mean')

            im = im.astype('uint16')

            imwrite(os.path.join(savefolder_registered, '1_CH488_000000.tif'), im)
            maxproj_xy = np.max(im, axis=0)
            maxproj_xz = np.max(im, axis=1)
            imwrite(os.path.join(savefolder_registered_max_xy, i_time + '.tif'),maxproj_xy)
            imwrite(os.path.join(savefolder_registered_max_xz, i_time + '.tif'),maxproj_xz)

            print("registered images")

        # =============================================================================
        # Preprocess data for segmentation (deconvolution, isotropic rescaling)
        # =============================================================================

        prepare_segmentations(im, savefolder_deconvolution, lateral_axialratio, psffilepath)

        try:
            import Segmentation.gradient_watershed.filters as grad_filters
        except:
            import gradient_watershed.filters as grad_filters
        import glob

        # re-load the deconvolved raws!
        rawfiles = glob.glob(os.path.join(savefolder_deconvolution,"*.tif"))
        im_deconvolved = skio.imread(rawfiles[0])
        print("loaded deconvolved images")

        # =============================================================================
        # generate binary image
        # =============================================================================

        im_binary = generate_binaryimage(im_deconvolved)

        # =============================================================================
        # perform connected component labeling and remove small labels
        # =============================================================================

        import skimage.measure as skmeasure
        connected_components_labels = skmeasure.label(im_binary>0)
        #remove labels that are smaller than 3000
        connected_components_labels_cleaned = grad_filters.remove_small_labels(connected_components_labels, min_size=3000)

        # =============================================================================
        # save image and visualization of segmentatino
        # =============================================================================

        labels_3D_connectedcomp_color = np.uint8(255 * skcolor.label2rgb(connected_components_labels_cleaned,
                                                                     colors=sns.color_palette('hls', n_colors=16),
                                                                     bg_label=0))

        #save image
        skio.imsave(os.path.join(savefolder_file,'labels_xy-connectedcomponentsRGB.tif'),
                    np.uint8(labels_3D_connectedcomp_color))
        skio.imsave(os.path.join(savefolder_file,'labels_xy-connectedcomponents.tif'),
                    np.uint16(connected_components_labels_cleaned))

