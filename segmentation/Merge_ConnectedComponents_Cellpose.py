import numpy as np
import os
from tifffile import imread, imwrite
from skimage.segmentation import relabel_sequential
import skimage.segmentation as sksegmentation
import skimage.color as skcolor
import seaborn as sns
import copy

class merge_segmentations():
    """
    This class merges the segmentation of the binary mask which is ideal for macrophages that do not touch and the
    cellpose based segmentation that is required to distinguish touching macrophages
    """

    def __init__(self):
        """
        init function and parameters
        """
        self.debug = 1
        self.segmentedimagename_cellpose = "labels_xy-integrated_gradients-correct_noexpand.tif"
        self.segmentedimagename_conn = "labels_xy-connectedcomponents.tif"
        self.maximalcellsize = 40000
        self.mincellsize=5000

    def process_timepoint(self, segmenteddata_connected, segmenteddata_cellpose, savefolder):
        """
        merge segmentations of one timepoint
        :param segmenteddata_connected: imagefile with segmented data of connected component
        :param segmenteddata_cellpose: imagefile with segmented data of cellpose segmentation
        :param savefolder: folder where to save merged, segmented data
        :return:
        """
        print(segmenteddata_connected)
        print(segmenteddata_cellpose)
        print(savefolder)

        # =============================================================================
        # open images
        # =============================================================================
        # segmenteddata_connected = os.path.join(segmenteddata_connected_folder, "high_stack_002", "t00008", "labels_xy-connectedcomponents.tif")
        # segmenteddata_cellpose = os.path.join(segmenteddata_cellpose_folder, "high_stack_002", "t00008", "labels_xy-integrated_gradients-correct_noexpand.tif")
        # savefolder = os.path.join(segmenteddata_cellpose_folder, "high_stack_002", "t00008", "test.tif")

        # open images
        segmentedimage_connected = imread(segmenteddata_connected)
        segmentedimage_cellpose = imread(segmenteddata_cellpose)

        merged_image = copy.deepcopy(segmentedimage_connected)

        # =============================================================================
        # replace labels that are too large in size in multi-otsu thresholded image with Cellpose based segmentation
        # =============================================================================

        #make binary 0 or 1
        im_binary_cellpose = segmentedimage_cellpose > 0.5

        #get maximum label in connected image
        maximumlabel = np.max(segmentedimage_connected)

        #process all labels
        for currentlabel in range(1, maximumlabel+1):
            #currentlabel = 2

            #get indices for specific label
            currentlabel_indices = np.where(segmentedimage_connected == currentlabel)

            #process all labels that are bigger than the maximalcellsize
            if currentlabel_indices[0].shape[0] > self.maximalcellsize:
                print("divide:" + str(currentlabel))
                merged_image[currentlabel_indices] =  im_binary_cellpose[currentlabel_indices] * (segmentedimage_cellpose[currentlabel_indices]+maximumlabel+1)

        #relabel image so that labels go from 1 to x
        relab, fw, inv = relabel_sequential(merged_image)

        # =============================================================================
        # merge small labels with close by macrophages
        # =============================================================================

        maximumlabel_relab = np.max(relab)

        # process all labels
        for currentlabel in range(1, maximumlabel_relab + 1):
            # currentlabel = 13

            #get indices for specific label
            currentlabel_indices = np.where(relab == currentlabel)
            if currentlabel_indices[0].shape[0] < self.mincellsize:
                currentcrop = relab[
                                         np.min(currentlabel_indices[0]):np.max(currentlabel_indices[0]),
                                         np.min(currentlabel_indices[1]):np.max(currentlabel_indices[1]),
                                         np.min(currentlabel_indices[2]):np.max(currentlabel_indices[2])]
                unique_label, unique_counts = np.unique(currentcrop, return_counts=True)
                sorted_counts = -np.sort(-unique_counts)
                newlabel=0
                for count_iter in range(len(sorted_counts)):
                    label = unique_label[unique_counts==(sorted_counts[count_iter])][0]
                    if label !=0 and label!=currentlabel:
                        newlabel = label
                        break
                print(newlabel)

                relab = np.where(relab == currentlabel,newlabel, relab)

        # =============================================================================
        # save merged image
        # =============================================================================

        # relabel image so that labels go from 1 to x
        relab2, fw2, inv2 = relabel_sequential(relab)

        imwrite(os.path.join(savefolder, 'labels_xy-merged.tif'), relab2)

        relab_color = np.uint8(255 * skcolor.label2rgb(relab2.copy(),
                                                                       colors=sns.color_palette('hls', n_colors=16),
                                                                       bg_label=0))

        imwrite(os.path.join(savefolder, 'labels_xy-connectedcomponentsRGB.tif'),
                    np.uint8(relab_color))



    def iterate_throughfolder(self, segmenteddata_connected_folder, segmenteddata_cellpose_folder, parentsavefolder):
        """
        go through parentfolder to process individual timepoints.
        :param segmenteddata_connected_folder: folder with the data of the connected components segmentation
        :param parentsavefolder:
        :return:
        """

        # get all timepoints from folder
        dir_list = os.listdir(segmenteddata_connected_folder)
        regionlist = []
        for path in dir_list:
            if path.startswith('high'):
                regionlist.append(path)
        regionlist.sort()
        print(regionlist)

        for region in regionlist:
            parentfolder_segmented = os.path.join(segmenteddata_connected_folder, region)
            # get all timepoints from folder
            dir_list = os.listdir(parentfolder_segmented)
            timepointlist = []
            for path in dir_list:
                if path.startswith('t'):
                    timepointlist.append(path)
            timepointlist.sort()
            print(timepointlist)

            # if you establish parameters, only open first timepoint
            if self.debug == 1:
                timepointlist = ["t00000"]

            for i_time in timepointlist:
                # if i_time<"t00069":
                # continue
                # construct filepath
                segmenteddata_connectedcomponents = os.path.join(parentfolder_segmented, i_time, self.segmentedimagename_conn)
                segmenteddata_cellpose = os.path.join(segmenteddata_cellpose_folder, region, i_time, self.segmentedimagename_cellpose)

                savefolder = os.path.join(parentsavefolder, region, i_time)

                try:
                    os.makedirs(savefolder)
                except OSError as error:
                    pass

                # process timepoint in region
                self.process_timepoint(segmenteddata_connectedcomponents, segmenteddata_cellpose, savefolder)



if __name__ == '__main__':

    #generate class function
    merge_it = merge_segmentations()

    parentfolder = "/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS"
    segmenteddata_cellpose_folder =  os.path.join(parentfolder, "Experiment0001_highresSeg_run_again", "processed_segmentation")
    segmenteddata_connected_folder = os.path.join(parentfolder, "Experiment0001_highresSeg_connectedComp_multiOtsu", "processed_segmentation_thres112")
    parentsavefolder = os.path.join(parentfolder, "Experiment0001_highresSeg_connectedComp_multiOtsu", "processed_segmentation_merged112")
    merge_it.debug = 0
    merge_it.iterate_throughfolder(segmenteddata_connected_folder, segmenteddata_cellpose_folder, parentsavefolder)

    segmenteddata_connected_folder = os.path.join(parentfolder, "Experiment0001_highresSeg_connectedComp_multiOtsu", "processed_segmentation_thres120")
    parentsavefolder = os.path.join(parentfolder, "Experiment0001_highresSeg_connectedComp_multiOtsu", "processed_segmentation_merged120")
    merge_it.debug = 0
    merge_it.iterate_throughfolder(segmenteddata_connected_folder, segmenteddata_cellpose_folder, parentsavefolder)

    segmenteddata_connected_folder = os.path.join(parentfolder, "Experiment0001_highresSeg_connectedComp_multiOtsu",
                                                  "processed_segmentation_thres125")
    parentsavefolder = os.path.join(parentfolder, "Experiment0001_highresSeg_connectedComp_multiOtsu",
                                    "processed_segmentation_merged125")
    merge_it.debug = 0
    merge_it.iterate_throughfolder(segmenteddata_connected_folder, segmenteddata_cellpose_folder, parentsavefolder)

    segmenteddata_connected_folder = os.path.join(parentfolder, "Experiment0001_highresSeg_connectedComp_multiOtsu",
                                                  "processed_segmentation_thres130")
    parentsavefolder = os.path.join(parentfolder, "Experiment0001_highresSeg_connectedComp_multiOtsu",
                                    "processed_segmentation_merged130")
    merge_it.debug = 0
    merge_it.iterate_throughfolder(segmenteddata_connected_folder, segmenteddata_cellpose_folder, parentsavefolder)


