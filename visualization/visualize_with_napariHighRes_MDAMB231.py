import numpy as np
import napari
import os

import skimage.transform
from tifffile import imread, imwrite

class image_visualizer():
    """
       This class generates visualizations of highres segmentation results.
       """

    def __init__(self):
        """
        init visualization tool
        """

        self.rawdatafolder = "/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/macrophage_control/20230602_Daetwyler_Xenograft/Experiment0013"
        experimentfolder_result = self.rawdatafolder + "_highresSeg"

        self.segmentationfolder = os.path.join(experimentfolder_result, 'result_segmentation', 'high_stack_001')
        self.visualizedfolder = os.path.join(experimentfolder_result, 'visualized')

        self.region = "high_stack_001"

        self.establish_param = 0

        # self.parentfolder = '/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/A375_WT/20220929_Daetwyler_Xenograft/Experiment0004results'
        # self.channel = "1_CH488_000000"
        # self.rawimage = "1_CH552_000000cp.tif"
        # self.labelimage = "1_CH552_000000sg.tif"


        self.viewer = napari.Viewer()


    def load_images(self, vis_param):
        """
        load images from folder and render them
        :return:
        """
        #get all timepoints from folder
        dir_list = os.listdir(self.rawdatafolder)
        timepointlist = []
        for path in dir_list:
            if path.startswith('t'):
                timepointlist.append(path)
        timepointlist.sort()
        print(timepointlist)
        #if you establish parameters, only open first timepoint
        if self.establish_param==1:
            timepointlist = ["t00000"]
            i_time="t00000"

        for i_time in timepointlist:

            #generate filepaths and folders
            rawimagepath_macrophages = os.path.join(self.rawdatafolder, i_time, self.region, vis_param['imagename_raw'])
            rawimagepath_cancer = os.path.join(self.rawdatafolder, i_time, self.region, vis_param['imagename_cancer'])

            visualization_folder1 = os.path.join(self.visualizedfolder, self.region, "angle_1a")
            visualization_folder2 = os.path.join(self.visualizedfolder, self.region, "angle_2a")
            try:
                os.makedirs(visualization_folder1)
            except OSError as error:
                pass
            try:
                os.makedirs(visualization_folder2)
            except OSError as error:
                pass

            visualized_file = os.path.join(visualization_folder1,  i_time + ".tif")
            visualized_file2 = os.path.join(visualization_folder2, i_time + ".tif")

            #open images
            input_image = imread(rawimagepath_macrophages)
            cancer_image= imread(rawimagepath_cancer)


            #add images as layers
            image_layer = self.viewer.add_image(input_image, gamma=vis_param['raw_gamma'], contrast_limits=vis_param['raw_contrast_limits'], colormap=vis_param['raw_colormap'])
            cancer_layer = self.viewer.add_image(cancer_image, gamma=vis_param['raw_gamma_cancer'], opacity=vis_param['opacity_cancer'], contrast_limits=vis_param['raw_contrast_limits_cancer'], colormap=vis_param['cancer_colormap'])

            #set rendering to 3D and set camera zoom parameters
            self.viewer.dims.ndisplay = vis_param['rendering_dimension']
            self.viewer.camera.zoom = vis_param['camera_zoom']

            #rescale 3D data to be correct dimensions
            self.viewer.layers['input_image'].scale = vis_param['raw_rescale_factor']
            self.viewer.layers['cancer_image'].scale = vis_param['raw_rescale_factor']

            #interpolation to cubic
            self.viewer.layers['input_image'].interpolation3d ='cubic'
            self.viewer.layers['cancer_image'].interpolation3d ='cubic'

            #self.viewer.layers['label_image_rescaled'].scale = vis_param['label_rescale_factor']
            self.viewer.layers.remove('cancer_image')

            #save a first camera position
            self.viewer.camera.angles = vis_param['camera_angle1']
            imagereturn = self.viewer.screenshot(canvas_only=True, scale=vis_param['scale_to_save'])
            imwrite(visualized_file, imagereturn)

            #save without vasculature
            #get angle from napari by entering: viewer.camera.angles in console
            self.viewer.camera.angles = vis_param['camera_angle2']
            #self.viewer.layers.remove('cancer_image')

            imagereturn2 = self.viewer.screenshot(canvas_only=True, scale=vis_param['scale_to_save'])
            imwrite(visualized_file2, imagereturn2)

            #if you establish the parameters, run napari, otherwise delete the layers for next timepoint
            if self.establish_param==1:
                napari.run()
            else:
                self.viewer.layers.remove('input_image')
                #self.viewer.layers.remove('cancer_image')




if __name__ == '__main__':

    visualization_param = dict(
        camera_angle1=(-179.3128327479437, 22.754843487827394, 94.38287509770109),
        camera_angle2=(-3.372776166910392, -1.3888899435527813, -105.76142446338508),
        camera_zoom=0.27,
        #raw_contrast_limits=(126,482),
        raw_contrast_limits=(87, 657),
        raw_contrast_limits_cancer=(102, 1529),
        #raw_gamma=0.67,
        raw_gamma=0.45,
        raw_gamma_cancer=0.59,
        raw_colormap ='gray_r',
        cancer_colormap='gray',
        opacity_cancer=0.58,
        opacity_label=1,
        rendering_dimension=3,
        label_blending='additive',
        # raw_rescale_factor =[9.210526, 1, 1],
        # label_rescale_factor = [9.210526, 1, 1],
        raw_rescale_factor =[2.564, 1, 1],
        label_rescale_factor =[2.564, 1, 1],
        #raw_rescale_factor=[1, 1, 1],
        #label_rescale_factor=[1, 1, 1],
        establish_param=0,
        set_label_colormap='default',
        scale_to_save=5,
        display_rawcancersignal=0,
        imagename_raw="1_CH488_000000.tif",
        imagename_cancer="1_CH594_000000.tif"

    )
    imagevisu = image_visualizer()

    imagevisu.rawdatafolder = "/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/MDA-MB-231/20230720_control/Experiment0010"
    experimentfolder_result = imagevisu.rawdatafolder + "_highres_visualized"
    imagevisu.segmentationfolder = os.path.join(experimentfolder_result, 'high_stack_001')
    imagevisu.visualizedfolder = os.path.join(experimentfolder_result, 'visualized_bright4')
    imagevisu.region = 'high_stack_002'
    imagevisu.establish_param = 0
    imagevisu.load_images(visualization_param)

    #adapt image
    pathfile = os.path.join(experimentfolder_result, "visualized_bright4", "angle_1a.tif")
    finalimage = imread(pathfile)

    # make boundingbox and bound image
    binarybox = np.zeros(finalimage.shape)
    binarybox[:, 812:2151, 348:3076,:] = 1

    pathfile_save = os.path.join(experimentfolder_result, "binarybox.tif")
    imwrite(pathfile_save, np.uint8(binarybox))



    finalimage[np.logical_and(finalimage==0, np.logical_not(binarybox))]=202

    print("test")



    selectedimage = finalimage + binarybox

    anotherstack_indices = np.where(selectedimage == 0)
    anotherstack_indices2 = np.where(finalimage == 0)

    finalimage[anotherstack_indices] = 202

    print("start file writing")
    pathfile_save = os.path.join(experimentfolder_result, "angle_1aV2.tif")
    imwrite(pathfile_save, np.uint8(finalimage))





