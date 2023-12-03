import pyclesperanto_prototype as cle
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.ndimage as ndimage

import numpy as np

print(cle.cl_info())

class segment_macrophage_lowres_class():
    """
    This class gathers all functions to segment the macrophage and cancer cells of the low res data.
    """

    def __init__(self):
        """
        initialization, including GPU initalization and parameter
        """
        #1. initialize GPU-------------------------------------
        gpu_devices = cle.available_device_names(dev_type="gpu")
        print("Available GPU OpenCL devices:" + str(gpu_devices))

        # selecting an Nvidia RTX
        cle.select_device("RTX")
        print("Using OpenCL device " + cle.get_device().name)
        self.image_tosegment = []
        self.lateral_axialratio = 3.5/0.38


    def open_image(self, imagename):
        """
        open image
        :param imagename: file path to current image
        :return:
        """
        self.image_tosegment = imread(imagename)

    def segment_macrophage(self, savepath, background_substracted_path="path", savepath_xlsx="pathxlsx"):
        """
        segment macrophages in low res image
        :param savepath: path to save segmentation outcome
        :param background_substracted_path: path to save background substracted image (optional)
        :param savepath_xlsx:  path to save segmentation statistics
        :return:
        """
        #downsample 3D volume so that it can be handled by GPU.
        im_rescaled = ndimage.zoom(self.image_tosegment, [1., 1. / self.lateral_axialratio, 1. / self.lateral_axialratio], order=1, mode='reflect')
        input_gpu = cle.push(im_rescaled)

        #pre-processing data

        background_subtracted = cle.top_hat_box(input_gpu, radius_x=6, radius_y=6, radius_z=6)
        if background_substracted_path != "path":
            imwrite(background_substracted_path, background_subtracted)
        print("background calculated")

        #thresholding and voronoi otsu labeling
        image1_t = cle.greater_constant(background_subtracted, None, 88.0)
        segmented = cle.voronoi_otsu_labeling(image1_t, None, 2.0, 1.0)
        segmented_withoutsmalllabels = cle.exclude_small_labels(segmented, maximum_size=10.0)

        segmented_array = cle.pull(segmented_withoutsmalllabels)

        #get segmentation statistics
        statistics = cle.statistics_of_labelled_pixels(input_gpu, segmented_array)
        table = pd.DataFrame(statistics)
        print("nb rows:" + str(table.shape[0]))
        if table.shape[0]<256:
            imwrite(savepath, segmented_array.astype("uint8"))
        else:
            imwrite(savepath, segmented_array.astype("uint16"))

        if savepath_xlsx != "pathxlsx":
            table.to_excel(savepath_xlsx)

    def segment_cancercells(self, savepath, background_substracted_path="path", savepath_xlsx="pathxlsx"):
        """
        segment cancer cells in low res image
        :param savepath: path to save segmentation outcome
        :param background_substracted_path: path to save background substracted image (optional)
        :param savepath_xlsx:  path to save segmentation statistics
        :return:
        """
        # pre-processing: mean box
        image1_denoised = cle.mean_box(self.input_gpu, None, 3.0, 3.0, 3.0)

        # pre-processing: top hat box filter
        background_subtracted = cle.top_hat_box(image1_denoised, None, 25.0, 25.0, 5.0)
        if background_substracted_path != "path":
            imwrite(background_substracted_path, background_subtracted)

        # segmentation: greater constant
        image3_gc = cle.greater_constant(background_subtracted, None, 87.0)
        # segmentation: apply voronoi otsu labeling to binary image.
        image4_vol = cle.voronoi_otsu_labeling(image3_gc, None, 7.0, 2.0)

        #remove labels from edge of image (artifacts)
        #excludeedge = cle.exclude_labels_on_edges(image4_vol)

        # post-processing: exclude small labels
        image5_E = cle.exclude_small_labels(image4_vol, maximum_size=5000.0)
        segmented_array = cle.pull(image5_E)

        #get segmentation statistics
        statistics = cle.statistics_of_labelled_pixels(self.input_gpu, segmented_array)
        table = pd.DataFrame(statistics)
        print("nb rows:" + str(table.shape[0]))
        if table.shape[0] < 256:
            imwrite(savepath, segmented_array.astype("uint8"))
        else:
            imwrite(savepath, segmented_array.astype("uint16"))

        if background_substracted_path != "pathxlsx":
            table.to_excel(savepath_xlsx)


    def show(self, image_to_show, labels=False):
        """
        This function generates three projections: in X-, Y- and Z-direction and shows them.
        """
        projection_x = cle.maximum_x_projection(image_to_show)
        projection_y = cle.maximum_y_projection(image_to_show)
        projection_z = cle.maximum_z_projection(image_to_show)

        fig, axs = plt.subplots(1, 3, figsize=(15, 15))
        cle.imshow(projection_x, plot=axs[0], labels=labels)
        cle.imshow(projection_y, plot=axs[1], labels=labels)
        cle.imshow(projection_z, plot=axs[2], labels=labels)

    def iterate_throughfolder(self, parentfolder, resultsfolder, channellist):
        """
        :param parentfolder: folder through which to iterate
        :param resultsfolder: folder where to save the outcome
        :param channellist: iterate through channels
        :return: segmented images will be saved in resultsfolder
        """

        dir_list = os.listdir(parentfolder)
        print(dir_list)
        timepointlist = []

        for path in dir_list:
            if path.startswith('t'):
                timepointlist.append(path)
        timepointlist.sort()

        print(timepointlist)

        for i_time in timepointlist:
            timepoint = i_time
            for i_channel in range(len(channellist)):

                #define filenames
                channel = channellist[i_channel]
                whichfile = channel + '.tif'
                whichfile_sg = channel + 'sg.tif'
                csvfilename = timepoint + ".xlsx"

                # construct filepath
                imagefilepath = os.path.join(parentfolder, timepoint, whichfile)
                print(imagefilepath)

                resultfilefolder = os.path.join(resultsfolder, channel, timepoint)
                resultfilepath_sg = os.path.join(resultfilefolder, whichfile_sg)
                resultfilepath_bg = os.path.join(resultfilefolder, "background_test.tif")
                csvfolderpath = os.path.join(resultsfolder + "_xlsx", channel)
                csvfilefolderpath = os.path.join(csvfolderpath, csvfilename)

                # make folders
                try:
                    os.makedirs(resultfilefolder)
                except OSError as error:
                    pass
                try:
                    os.makedirs(csvfolderpath)
                except OSError as error:
                    pass

                # open image
                self.open_image(imagefilepath)

                # segment
                if channel == '1_CH488_000000':
                    self.segment_macrophage(resultfilepath_sg, savepath_xlsx=csvfilefolderpath, background_substracted_path=resultfilepath_bg)
                else:
                    self.segment_cancercells(resultfilepath_sg, savepath_xlsx=csvfilefolderpath)



if __name__ == '__main__':
    #init low res segmentation class
    segment_class = segment_macrophage_lowres_class()
    #parameters-----------------------------------
    parentfolder = '/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS/Experiment0001_stitched/fish1'
    parentfolder = '/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/macrophage_control/20230602_Daetwyler_Xenograft/Experiment0013_stitched/fish3'
    resultsfolder = parentfolder + "_segmented"

    channellist = ['1_CH488_000000']
    #perform actions
    segment_class.iterate_throughfolder(parentfolder, resultsfolder, channellist)


