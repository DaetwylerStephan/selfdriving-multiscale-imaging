import copy

import numpy as np
import os
from tifffile import imread, imwrite
import skimage.color as skcolor
import seaborn as sns
from skimage.segmentation import relabel_sequential

class manual_curate_segmentation():
    """
    This class provides functions to manually curate images
    """

    def remove_label(self, stack, label):
        """
        removes a specific label from a stack
        :param stack: numpy array
        :param label: label
        :return: processed numpy array
        """
        currentlabel_indices = np.where(stack == label)
        stack[currentlabel_indices] = 0
        return stack

    def merge_label(self, stack, label1, label2):
        """
        merge labels (replaces label2 with label1)
        :param stack: numpy array to process
        :param label1: integer label
        :param label2: integer label
        :return: processed numpy array
        """
        currentlabel_indices = np.where(stack == label2)
        stack[currentlabel_indices] = label1
        return stack

    def takeLabel_fromanotherstack(self, stack, anotherstack, label):
        """
        take a label from another stack (anotherstack) and insert it into the current stack (stack)
        :param stack: numpy array to process
        :param anotherstack: numpy array to take label from
        :param label: label to take
        :return: processed numpy array
        """

        anotherstack_indices = np.where(anotherstack == label)
        maximumlabel = np.max(stack)
        newlabel = anotherstack[anotherstack_indices] + maximumlabel + 1
        stack[anotherstack_indices] = anotherstack[anotherstack_indices] + maximumlabel + 1
        return stack, newlabel[0]

    def takeLabel_fromanotherstack_range(self, stack, anotherstack_orig, label, xrange=[], yrange=[], zrange=[]):
        """
        take a label from another stack (anotherstack) and insert it into the current stack (stack)
        :param stack: numpy array to process
        :param anotherstack: numpy array to take label from
        :param label: label to take
        :return: processed numpy array
        """
        anotherstack = copy.deepcopy(anotherstack_orig)
        if not zrange:
            zrange = [0, anotherstack.shape[0]]
        if not xrange:
            xrange = [0, anotherstack.shape[1]]
        if not yrange:
            yrange = [0, anotherstack.shape[2]]

        #make boundingbox and bound image
        binarybox = np.zeros(anotherstack.shape)
        binarybox[zrange[0]:zrange[1], xrange[0]:xrange[1], yrange[0]:yrange[1]]=1

        selectedimage = binarybox * anotherstack

        #select only specific labels
        anotherstack_indices = np.where(selectedimage == label)

        maximumlabel = np.max(stack)

        newlabel = anotherstack[anotherstack_indices] + maximumlabel + 1
        stack[anotherstack_indices] = anotherstack_orig[anotherstack_indices] + maximumlabel + 1

        return stack, newlabel[0]

    def replace_one_label(self, stack, anotherstack, labeltoreplace):

        # make binary 0 or 1
        im_binary = anotherstack > 0.5
        currentlabel_indices = np.where(stack == labeltoreplace)
        maximumlabel = np.max(stack)

        stack[currentlabel_indices] = im_binary[currentlabel_indices] * (anotherstack[currentlabel_indices] + maximumlabel + 1)
        return stack

    def give_label_new_identiyinarea(self, stack, label, xrange=[], yrange=[], zrange=[]):

        anotherstack = copy.deepcopy(stack)
        if not zrange:
            zrange = [0, anotherstack.shape[0]]
        if not xrange:
            xrange = [0, anotherstack.shape[1]]
        if not yrange:
            yrange = [0, anotherstack.shape[2]]

        # make boundingbox and bound image
        binarybox = np.zeros(anotherstack.shape)
        binarybox[zrange[0]:zrange[1], xrange[0]:xrange[1], yrange[0]:yrange[1]] = 1

        selectedimage = binarybox * anotherstack

        # select only specific labels
        anotherstack_indices = np.where(selectedimage == label)

        maximumlabel = np.max(stack)

        newlabel = anotherstack[anotherstack_indices] + maximumlabel + 1
        stack[anotherstack_indices] = stack[anotherstack_indices] + maximumlabel + 1

        return stack, newlabel[0]

    def saveimage(self, new_image, path_save, path_save_RGB):
        # relabel image so that labels go from 1 to x
        relab, fw2, inv2 = relabel_sequential(new_image)
        # save new curated
        imwrite(path_save, relab)
        color_image = np.uint8(255 * skcolor.label2rgb(relab.copy(),
                                                       colors=sns.color_palette('hls', n_colors=16),
                                                       bg_label=0))
        imwrite(path_save_RGB, np.uint8(color_image))





if __name__ == "__main__":
    # =============================================================================
    # initializations
    # =============================================================================
    region = "high_stack_002"
    timepoint = "t00049"
    parentfolder = "/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS"
    data_segmentation = os.path.join(parentfolder, "Experiment0001_highresSeg_connectedComp_multiOtsu", "processed_segmentation_merged130", region, timepoint, "labels_xy-merged.tif")
    data_before_processing = os.path.join(parentfolder, "Experiment0001_highresSeg_connectedComp_multiOtsu", "result_segmentation", region, timepoint, "labels_xy-connectedcomponents.tif")
    data_cellpose =  os.path.join(parentfolder, "Experiment0001_highresSeg_run_again", "result_segmentation", region, timepoint, "labels_xy-integrated_gradients-correct_noexpand.tif")
    data_segmentation = os.path.join(parentfolder, "Experiment0001_highres_manuallyCompiled",  region, timepoint, "labels_xy-merged.tif")

    path_save_folder = os.path.join(parentfolder, "Experiment0001_highres_manuallyCompiled2", region, timepoint)
    path_save = os.path.join(path_save_folder, "labels_xy-merged.tif")
    path_save_RGB = os.path.join(path_save_folder, "labels_xy-merged_componentsRGB.tif")
    try:
        os.makedirs(path_save_folder)
    except OSError as error:
        pass


    """parameters"""
    curate_it = manual_curate_segmentation()

    new_image = imread(data_segmentation)
    multiOtsu_segmentation = imread(data_before_processing)
    cellpose_segmentation = imread(data_cellpose)

    # new_image, newlabel = curate_it.give_label_new_identiyinarea(new_image, 8, yrange=[187, 390])
    # new_image, newlabel = curate_it.takeLabel_fromanotherstack(new_image, cellpose_segmentation, 78)
    #
    # new_image, newlabel = curate_it.give_label_new_identiyinarea(new_image, 13, yrange=[0, 158])
    #
    # new_image, newlabel2 = curate_it.give_label_new_identiyinarea(new_image, 13, xrange=[129, 280])
    new_image = curate_it.merge_label(new_image, 16, 17)

    # example t00003
    # new_image = curate_it.replace_one_label(new_image, cellpose_segmentation, 1)
    # new_image = curate_it.remove_label(new_image, 21)
    # new_image, newlabel = curate_it.takeLabel_fromanotherstack(new_image, cellpose_segmentation, 4)
    # new_image = curate_it.merge_label(new_image, 16, 17)
    # new_image, newlabel = curate_it.takeLabel_fromanotherstack(new_image, multiOtsu_segmentation,12)
    # new_image, newlabel = curate_it.takeLabel_fromanotherstack(new_image, multiOtsu_segmentation,19)
    # new_image, newlabel = curate_it.takeLabel_fromanotherstack(new_image, multiOtsu_segmentation,9)
    # new_image, newlabel = curate_it.takeLabel_fromanotherstack_range(new_image, cellpose_segmentation,61, zrange=[78,170])
    # new_image = curate_it.merge_label(new_image, 20, newlabel)
    #
    # t000023: new_image, newlabel = curate_it.give_label_new_identiyinarea(new_image, 10, yrange=[437,(501)], xrange=[244,279])
    # new_image, newlabel = curate_it.takeLabel_fromanotherstack_range(new_image, multiOtsu_segmentation, 2,zrange=[50,75],
    #                                                                 yrange=[223, 288], xrange=[120, 155])


    # save image
    curate_it.saveimage(new_image, path_save, path_save_RGB)



    # new_image = imread(path_save)
    # new_image=curate_it.merge_label(new_image, 18,20)


    # example t00027
    # new_image = curate_it.merge_label(new_image, 17, 20)
    # new_image = curate_it.merge_label(new_image, 14, 26)
    # new_image = curate_it.merge_label(new_image, 21, 22)
    # new_image, newlabel = curate_it.takeLabel_fromanotherstack(new_image, cellpose_segmentation, 100)
    # new_image = curate_it.merge_label(new_image, 24, newlabel)
    # new_image = curate_it.replace_one_label(new_image, cellpose_segmentation, 23)


