
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


def generate2Dsegmentations(im, im_cancer, savedeconvfilepath):
    """
    generate preprocessed/deconvolved data and cell pose segmentation masks

    :param im: image of macrophages
    :param im_cancer: image of cancer cells
    :param im_cancer: folder to save the deconvolved file
    :return: saved images and cellpose segmentation masks
    """
    # =============================================================================
    # Resample to isotropic
    # =============================================================================

    im = ndimage.zoom(im, [1., 1. / lateral_axialratio, 1. / lateral_axialratio], order=1, mode='reflect')
    im_cancer = ndimage.zoom(im_cancer, [1., 1 / lateral_axialratio, 1 / lateral_axialratio], order=1, mode='reflect')
    print("resampled images")

    # =============================================================================
    #     Preprocess fully in 3D
    # =============================================================================

    # rescale
    # im = im_preprocessing.normalize(im, pmin=2, pmax=99.8, clip=True)
    # im_cancer = im_preprocessing.normalize(im_cancer, pmin=2, pmax=99.8, clip=True)

    # smooth background
    im_bg = im_preprocessing.smooth_vol(im, ds=5, smooth=5)
    im_cancer_bg = im_preprocessing.smooth_vol(im_cancer, ds=5, smooth=5)

    im = im_preprocessing.normalize(im - im_bg, pmin=2, pmax=99.8, clip=True)
    im_cancer = im_preprocessing.normalize(im_cancer - im_cancer_bg, pmin=2, pmax=99.8, clip=True)

    print("normalized")
    # deconvolution
    psf_meSPIM = spio.loadmat(psffilepath)['PSF']
    psf_meSPIM = psf_meSPIM / (float(np.sum(psf_meSPIM)))

    im_deconv = skrestoration.wiener(im, psf_meSPIM,
                                     balance=0.5)  # was doing balance=0.5 # use a smaller balance to retain sharper features.
    im_deconv = np.clip(im_deconv, 0, 1)
    im_deconv = np.array(
        [im_preprocessing.anisodiff(ss, niter=15, kappa=1, gamma=0.1, step=(1., 1.), sigma=0, option=1, ploton=False)
         for ss in im_deconv])

    im_cancer_deconv = skrestoration.wiener(im_cancer, psf_meSPIM,
                                            balance=0.5)  # was doing balance=0.5 # use a smaller balance to retain sharper features.
    im_cancer_deconv = np.clip(im_cancer_deconv, 0, 1)
    im_cancer_deconv = np.array(
        [im_preprocessing.anisodiff(ss, niter=15, kappa=1, gamma=0.1, step=(1., 1.), sigma=0, option=1, ploton=False)
         for ss in im_cancer_deconv])
    print("deconvolved")

    # demix_videos(vid1, vid2, l1_ratio=0.5)
    im_unmix = im_preprocessing.demix_videos(im_deconv, im_cancer_deconv, l1_ratio=0.5)

    # rescale.
    im_unmix_1 = im_preprocessing.normalize(im_unmix[..., 0], clip=True)
    im_unmix_2 = im_preprocessing.normalize(im_unmix[..., 1], clip=True)

    skio.imsave(os.path.join(savedeconvfilepath, basename + 'deconv_demix.tif'),
                np.uint16(65535 * im_unmix_1))

    # =============================================================================
    # Cellpose segmentation
    # =============================================================================
    from cellpose import models, core

    # set up Cellpose parameters
    use_GPU = core.use_gpu()
    modelname = 'cyto2'
    print(modelname)
    model = models.Cellpose(model_type=modelname, gpu=True);  # default is CPU!
    channels = [0, 0]

    # angle = 'xy'
    # angle = 'xz'
    # angle = 'yz'  # seems ok with the smaller image.
    anglelist = ['xy', 'xz', 'yz']

    im_unmix_orig = im_unmix_1.copy()

    for angle in anglelist:
        print("angle " + angle + " is segmented")

        if angle == 'xy':
            im_unmix_1 = im_unmix_orig.copy()
        if angle == 'xz':
            im_unmix_1 = im_unmix_orig.transpose(1, 0, 2).copy()
        if angle == 'yz':
            im_unmix_1 = im_unmix_orig.transpose(2, 0, 1).copy()


        # fig, ax = plt.subplots()
        # plt.title('midslice')
        # plt.imshow(im_unmix_1[im_unmix_1.shape[0] // 2] + 10)
        # plt.show()

        all_masks = []
        # this is still quicker!
        for dd in tqdm(np.arange(len(im_unmix_1))):
            img = im_unmix_1[dd].copy()
            imgs = [img]

            masks, flows, styles, diams = model.eval(imgs,  # if not a list doesn't run slice by slice !
                                                     batch_size=32,  # this parameter does nothing.....
                                                     channels=channels,
                                                     do_3D=False,
                                                     # diameter=100, # will need to run at multiple scales!.
                                                     diameter=30,  # was used for the other views.
                                                     net_avg=False,
                                                     model_loaded=True,
                                                     progress=True)

            all_masks.append(masks[0])

            # plt.figure()
            # plt.imshow(imgs[0])
            # plt.show()
            #
            # plt.figure()
            # plt.imshow(masks[0])
            # plt.show()

        # =============================================================================
        # save image
        # =============================================================================

        savedict = {'masks': all_masks, 'angle': angle}

        # save as a numpy?
        try:
            spio.savemat(os.path.join(savefolder_file, basename + '_cellpose_%s.mat' % (angle)),
                         savedict, do_compression=True)
            print("saving success")
        except:

            import pickle

            savepicklefile = os.path.join(savefolder_file, basename + '_cellpose_%s.pickle' % (angle))
            with open(savepicklefile, 'wb') as handle:
                pickle.dump(savedict, handle)  # , protocol=pickle.HIGHEST_PROTOCOL)


def generate_binaryimage(im_deconvolved):
    """
    generate binary image from deconvolved image using otsu tresholding
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
    binary_thresh = skfilters.threshold_otsu(im_)  # what the best threshold?  # this is bad
    im_binary = im_ > binary_thresh  # +0.1;

    im_binary = skmorph.remove_small_objects(im_binary, min_size=1000, connectivity=2)

    comb = np.maximum(im_binary * 1, (dog - dog.mean()) / (4 * np.std(dog)))
    im_binary = comb >= 1
    im_binary = skmorph.binary_closing(im_binary, skmorph.ball(1))
    im_binary = ndimage.binary_fill_holes(im_binary)

    # plt.figure()
    # plt.imshow(im_binary[20])
    # plt.figure()
    # plt.imshow(mask_xy[20])
    # plt.show()

    skio.imsave(os.path.join(savefolder_file, basename + 'im_binary.tif'),
                np.uint16(im_binary))
    print("binary mask generated")
    return im_binary

def merge_masks(mask_xy, mask_xz, mask_yz, im_binary):
    """
    return the gradients fused from 3 orthogonal segmentations

    :param mask_xy: Cellpose segmentation in xy direction
    :param mask_xz: Cellpose segmentation in xz direction
    :param mask_yz: Cellpose segmentation in yz direction
    :return: labels gradients for 3D watershed
    """
    # im_binary_dtform = ndimage.distance_transform_edt(im_binary)  # this works.

    # don't expand the mask.
    # mask_xy = grad_filters.expand_masks2D(mask_xy, binary=im_binary, dist_tform=im_binary_dtform)   #### looks like use this binary is hm....
    # mask_xz = grad_filters.expand_masks2D(mask_xz, binary=im_binary.transpose(1,0,2), dist_tform=im_binary_dtform.transpose(1,0,2))
    # mask_yz = grad_filters.expand_masks2D(mask_yz, binary=im_binary.transpose(2,1,0), dist_tform=im_binary_dtform.transpose(2,1,0))

    # mask_xy
    guide_image_xy = grad_flows.distance_transform_labels_fast(mask_xy)
    mask_xy_gradient = grad_flows.distance_centroid_tform_flow_labels2D_parallel(mask_xy,
                                                                                 dtform_method='cellpose_improve',
                                                                                 guide_image=guide_image_xy,
                                                                                 fixed_point_percentile=0.05,
                                                                                 n_processes=32,
                                                                                 power_dist=0.5)

    print(mask_xy_gradient[:, 0].min(), mask_xy_gradient[:, 0].max())
    mask_xy_gradient = mask_xy_gradient[:, 1:].copy()
    mask_xy_gradient = np.concatenate([np.zeros_like(mask_xy_gradient[:, 1])[:, None, ...], mask_xy_gradient], axis=1)
    mask_xy_gradient = mask_xy_gradient[:, [0, 1, 2], ...].copy()  # we must flip the channels!.

    #mask_xz
    guide_image_xz = grad_flows.distance_transform_labels_fast(mask_xz)
    mask_xz_gradient = grad_flows.distance_centroid_tform_flow_labels2D_parallel(mask_xz,
                                                                                 dtform_method='cellpose_improve',
                                                                                 guide_image=guide_image_xz,
                                                                                 fixed_point_percentile=0.05,
                                                                                 n_processes=32,
                                                                                 power_dist=0.5)
    mask_xz_gradient = mask_xz_gradient[:, 1:].copy()
    mask_xz_gradient = mask_xz_gradient.transpose(2, 1, 0, 3)
    mask_xz_gradient = np.concatenate([np.zeros_like(mask_xz_gradient[:, 1])[:, None, ...], mask_xz_gradient], axis=1)
    mask_xz_gradient = mask_xz_gradient[:, [1, 0, 2], ...].copy()  # we must flip the channels!.

    #mask_yz
    guide_image_yz = grad_flows.distance_transform_labels_fast(mask_yz)
    mask_yz_gradient = grad_flows.distance_centroid_tform_flow_labels2D_parallel(mask_yz,
                                                                                 dtform_method='cellpose_improve',
                                                                                 guide_image=guide_image_yz,
                                                                                 fixed_point_percentile=0.05,
                                                                                 n_processes=32,
                                                                                 power_dist=0.5)
    mask_yz_gradient = mask_yz_gradient[:, 1:].copy()
    mask_yz_gradient = mask_yz_gradient.transpose(2, 1, 3, 0)
    #mask_yz_gradient = mask_yz_gradient.transpose(3, 1, 2, 0)
    mask_yz_gradient = np.concatenate([np.zeros_like(mask_yz_gradient[:, 1])[:, None, ...], mask_yz_gradient], axis=1)
    mask_yz_gradient = mask_yz_gradient[:,[1,2,0],...].copy() # we must flip the channels!.

    dx = grad_filters.var_combine([ndimage.gaussian_filter(mask_xy_gradient[:, 2], sigma=1),
                                   ndimage.gaussian_filter(mask_xz_gradient[:, 2], sigma=1)],
                                  ksize=5,
                                  alpha=0.5)
    dy = grad_filters.var_combine([ndimage.gaussian_filter(mask_xy_gradient[:, 1], sigma=1),
                                   ndimage.gaussian_filter(mask_yz_gradient[:, 1], sigma=1)],
                                  ksize=5,
                                  alpha=0.5)
    dz = grad_filters.var_combine([ndimage.gaussian_filter(mask_xz_gradient[:, 0], sigma=1),
                                   ndimage.gaussian_filter(mask_yz_gradient[:, 0], sigma=1)],
                                  ksize=5,
                                  alpha=0.5)

    # # does this change vs
    dx = ndimage.gaussian_filter(dx, sigma=1.)
    dy = ndimage.gaussian_filter(dy, sigma=1.)
    dz = ndimage.gaussian_filter(dz, sigma=1.)
    labels_gradients = np.concatenate([dz[:, None, ...],
                                       dy[:, None, ...],
                                       dx[:, None, ...]], axis=1)
    # do a little filtering.
    # labels_gradients = np.array([ndimage.gaussian_filter(labels_gradients[:,ch], sigma=1) for ch in np.arange(labels_gradients.shape[1])]) # should only be 3 channels.
    labels_gradients = labels_gradients.transpose(1, 0, 2, 3)

    print(labels_gradients.shape)
    # this normalizes!
    labels_gradients = labels_gradients / (
                np.linalg.norm(labels_gradients, axis=0)[None, ...] + 1e-20)  # this is good. !
    print(np.linalg.norm(labels_gradients, axis=0).max())  # verify this is unit magnitude!!!

    # =============================================================================
    #     reshaping for input to grad watershed!.
    # =============================================================================
    # this corrects for the shape to be inputted!. holy cow.
    labels_gradients = labels_gradients.transpose(1, 2, 3, 0).astype(np.float32)

    print(labels_gradients.shape)

    # plt.figure()
    # plt.imshow(labels_gradients[0, ..., 1])
    # plt.show()
    # wtf?
    # labels_gradients[im_binary==0] = 0 # setting all gradients outside mask to  be 0 ....
    print(labels_gradients.shape)

    return labels_gradients


def register_stack(imagestack, applymedian=0, reference_flag='mean'):
    """
    Register the stack to account for sample drift that occurs
    :param imagestack: stack to register
    :param applymedian: apply value to zero elements if non zero
    :param reference_flag: how should stackreg register stack
    :return: registered stack
    """

    # calculate median value
    if applymedian == 0:
        applymedian = np.median(imagestack[0, :, :])

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
    # initializations
    # =============================================================================
    """parameters"""
    lateral_axialratio = 3.42 #for (down-)sampling image to isotropic voxels, e.g. 3.42 for axial spacing of 0.4 um and 0.117 nm lateral; or 2.56 for 0.3
    debug = 0
    region = "high_stack_001"
    alignstackbefore = 1 #do you want to align/register stack before segmentation? (use pystackreg)

    experimentfolder = "/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS/Experiment0001"
    experimentfolder_result = experimentfolder + "_highresSeg"
    # PSF file path
    psffilepath = "/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/psf/meSPIM_PSF_kernel.mat"


    """init Image-preprocessing class"""
    try:
        from Segmentation.preprocessing_class import ImagePreprocessing as im_preprocess
    except:
        from preprocessing_class import ImagePreprocessing as im_preprocess
    im_preprocessing = im_preprocess()

    # =============================================================================
    # Load the images and generate save folder
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
        timepointlist = ['t00021']

    for i_time in timepointlist:

        # if i_time > 't00017':
        #     continue  # continue here

        print(i_time)

        # construct filepaths
        imagefilepath = os.path.join(experimentfolder, i_time, region,'1_CH488_000000.tif')
        file_cancer = os.path.join(experimentfolder, i_time, region, '1_CH552_000000.tif')
        basename = os.path.split(imagefilepath)[-1].split('.tif')[0]
        print(imagefilepath)

        savefolder_deconvolution = os.path.join(experimentfolder_result, "deconvolution", region, i_time)
        savefolder_file = os.path.join(experimentfolder_result, "result_segmentation", region, i_time)

        #make folders
        im_preprocessing.mkdir(savefolder_deconvolution)
        im_preprocessing.mkdir(savefolder_file)


        im = skio.imread(imagefilepath)
        im_cancer = skio.imread(file_cancer)


        print("loaded images")

        # =============================================================================
        # 3D register stacks if there was a lot of movement / jitters
        # =============================================================================

        if alignstackbefore==1:
            print("register stack plane by plane")




        # =============================================================================
        # Generate and save 2D segmentations with Cellpose
        # =============================================================================

        generate2Dsegmentations(im, im_cancer, savefolder_deconvolution)

        # =============================================================================
        # now merge the different segmentations
        # =============================================================================

        try:
            import Segmentation.gradient_watershed.filters as grad_filters
            import Segmentation.gradient_watershed.watershed as grad_watershed
            import Segmentation.gradient_watershed.flows as grad_flows

        except:
            import gradient_watershed.filters as grad_filters
            import gradient_watershed.watershed as grad_watershed
            import gradient_watershed.flows as grad_flows


        import glob
        #read the segmentation
        files = glob.glob(os.path.join(savefolder_file,"*.mat"))
        mask_xy = spio.loadmat(files[0])['masks']; mask_xy = grad_filters.relabel_slices(mask_xy); mask_xy = grad_filters.filter_segmentations_axis(mask_xy, window=3, min_count=5)
        mask_xz = spio.loadmat(files[1])['masks']; mask_xz = grad_filters.relabel_slices(mask_xz); mask_xz = grad_filters.filter_segmentations_axis(mask_xz, window=3, min_count=5)
        mask_yz = spio.loadmat(files[2])['masks']; mask_yz = grad_filters.relabel_slices(mask_yz); mask_yz = grad_filters.filter_segmentations_axis(mask_yz, window=3, min_count=5)
        print("masks loaded")

        # plt.figure()
        # plt.imshow(mask_xy[35])
        # plt.show()

        # read the deconvolved raws!
        rawfiles = glob.glob(os.path.join(savefolder_deconvolution,"*.tif"))
        im_deconvolved = skio.imread(rawfiles[0])

        print("loaded deconvolved images")

        # =============================================================================
        # generate binary image
        # =============================================================================

        im_binary = generate_binaryimage(im_deconvolved)

        # =============================================================================
        # generate the gradients from the Cellpose segmentation to identify cell identities
        # =============================================================================

        labels_gradients = merge_masks(mask_xy, mask_xz, mask_yz, im_binary)

        # =============================================================================
        # perform watershed segmentation within the binary segmentation
        # =============================================================================

        labels_3D_watershed, cell_seg_connected, tracks, votes_grid_acc = grad_watershed.gradient_watershed3D_binary(
            im_binary > 0,
            gradient_img=labels_gradients,
            divergence_rescale=False,
            # i don't think this was defined correct helps.... ----> this actually seems to help by putting random fiducials...... (not actually stable..)
            smooth_sigma=1,  # use a larger for weird shapes!.
            smooth_gradient=1,  # this has no effect except divergence_rescale=True
            delta=1.,  # kept at 1 for 1px hop.
            n_iter=50,  # evolve more - the limit isn't very stable (median seems to be more stable. )
            eps=1e-12,
            min_area=5,
            thresh_factor=None,
            mask=im_binary,
            debug_viz=False)  # 50 seems not enough for 3D... as it is very connected.

        # remove small labels
        labels_3D_watershed = labels_3D_watershed * im_binary
        labels_3D_watershed = grad_filters.remove_small_labels(labels_3D_watershed, min_size=1000)  # this removed ?

        # remove planar segmentations that are artifacts of 2D.
        print(np.unique(labels_3D_watershed))

        labels_3D_watershed_color = np.uint8(255 * skcolor.label2rgb(labels_3D_watershed,
                                                                     colors=sns.color_palette('hls', n_colors=16),
                                                                     bg_label=0))
        # bg_color=(1,1,1))) # this was put as white..... ? # that shudn't save well.

        skio.imsave(os.path.join(savefolder_file,
                                 'labels_xy-integrated_gradients-RGB-correct_noexpand.tif'),
                    np.uint8(labels_3D_watershed_color))
        skio.imsave(os.path.join(savefolder_file,
                                 'labels_xy-integrated_gradients-correct_noexpand.tif'),
                    np.uint16(labels_3D_watershed))


