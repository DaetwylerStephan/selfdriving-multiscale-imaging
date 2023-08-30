
import numpy as np
import napari
import os
import cupy as cp
import skimage.transform
import cucim.skimage.transform
from tifffile import imread, imwrite

class image_visualizer_stitched():
    """
       This class generates visualizations of lowres macrophage segmentation results.
       """

    def __init__(self):
        """
        init visualization tool
        """

        #data folders
        rawdatafolder_parent = "folderpath"
        self.rawdatafolder = os.path.join(rawdatafolder_parent, "fish1") #rawdata
        self.segmentationfolder = os.path.join(rawdatafolder_parent, "fish1_segmented", "1_CH488_000000") #segmented data
        self.visualizedfolder = self.rawdatafolder + "_visualized" #where to save visualized data

        #start napari viewer
        self.viewer = napari.Viewer()

    def load_visualize_images(self, vis_param):
        """
        load images from folder and render them
        :param vis_param: Python dict of visulization param
                        *vis_param['camera_angle1']
        :return:
        """

        # get all timepoints from folder
        dir_list = os.listdir(self.segmentationfolder)
        timepointlist = []
        for path in dir_list:
            if path.startswith('t'):
                timepointlist.append(path)
        timepointlist.sort()

        # if you establish parameters, only open first timepoint
        if vis_param['establish_param'] == 1:
            timepointlist = ["t00000"]

        for i_time in timepointlist:

            # generate filepaths and folders
            segmentedimagepath = os.path.join(self.segmentationfolder, i_time, "1_CH488_000000sg.tif")
            segmentedimagepath_red = os.path.join(self.segmentationfolder, i_time, "1_CH488_000000sg_red.tif")
            segmentedimagepath_blue = os.path.join(self.segmentationfolder, i_time, "1_CH488_000000sg_blue.tif")
            segmentedimagepath_green = os.path.join(self.segmentationfolder, i_time, "1_CH488_000000sg_green.tif")

            rawimagepath = os.path.join(self.rawdatafolder, i_time, "1_CH488_000000.tif")
            rawimagepath_vessels = os.path.join(self.rawdatafolder, i_time, "1_CH594_000000.tif")

            visualization_folder1 = os.path.join(self.visualizedfolder,  "angle_1a")
            visualization_folder2 = os.path.join(self.visualizedfolder, "angle_2a")

            try:
                os.makedirs(visualization_folder1)
            except OSError as error:
                pass
            try:
                os.makedirs(visualization_folder2)
            except OSError as error:
                pass

            visualized_file = os.path.join(visualization_folder1, i_time + ".tif")
            visualized_file2 = os.path.join(visualization_folder2, i_time + ".tif")

            # open images
            input_image_raw = imread(rawimagepath)
            input_image_vessel_raw = imread(rawimagepath_vessels)
            label_image = imread(segmentedimagepath)
            label_image_red = imread(segmentedimagepath_red)
            label_image_blue = imread(segmentedimagepath_blue)
            label_image_green = imread(segmentedimagepath_green)

            print(input_image_raw.shape)

            #downsample raw images to same size as label images
            #if gpu is not big enoug, use: label_image_rescaled = skimage.transform.resize(label_image, input_image.shape, order=0)

            import time
            start_time = time.time()
            cp._default_memory_pool.free_all_blocks()
            input_image_gpu = cp.array(input_image_raw)
            input_image_gpu = cucim.skimage.transform.resize(input_image_gpu, label_image.shape, order=0)
            input_image = np.asarray(input_image_gpu.get())
            del input_image_gpu

            cp._default_memory_pool.free_all_blocks()
            input_image_gpu = cp.array(input_image_vessel_raw)
            input_image_gpu = cucim.skimage.transform.resize(input_image_gpu, label_image.shape, order=0)
            input_image_vessels = np.asarray(input_image_gpu.get())
            del input_image_gpu

            print("--- %s seconds ---" % (time.time() - start_time))
            print(label_image.shape)


            # alternatively, use opencv if file too big for GPU:

            # start_time = time.time()
            # import numpy as np
            # input_image = np.zeros((label_image.shape[0], label_image.shape[1], label_image.shape[2]))
            # for idx in range(label_image.shape[0]):
            #     img = input_image_raw[idx, :, :]
            #     img_sm = cv2.resize(img, (label_image.shape[2], label_image.shape[1]), interpolation=cv2.INTER_NEAREST)
            #     input_image[idx, :, :] = img_sm
            # print("--- %s seconds ---" % (time.time() - start_time))

            print(input_image.shape)
            print(label_image.shape)

            # add images as layers
            image_layer = self.viewer.add_image(input_image, gamma=vis_param['raw_gamma'], contrast_limits=vis_param['raw_contrast_limits'])
            image_layer_vessels = self.viewer.add_image(input_image_vessels, gamma=vis_param['raw_gamma_vessel'], contrast_limits=vis_param['raw_contrast_limits_vessels'], blending=vis_param['label_blending'])
            layer_image_rescaled = self.viewer.add_labels(label_image, opacity=vis_param['label_gamma'], blending=vis_param['label_blending'])
            layer_image_red = self.viewer.add_image(label_image_red, opacity=vis_param['label_gamma'], blending='additive', colormap='red')
            layer_image_blue = self.viewer.add_image(label_image_blue, opacity=vis_param['label_gamma'], blending='additive',colormap='blue')
            layer_image_green = self.viewer.add_image(label_image_green, opacity=vis_param['label_gamma'], blending='additive', colormap='green')
            layer_image_green.visible='true' #interesting hack to make it display colors

            # set rendering to 3D and set camera zoom parameters
            self.viewer.dims.ndisplay = vis_param['rendering_dimension']
            self.viewer.camera.zoom = vis_param['camera_zoom']

            # rescale 3D data to be correct dimensions
            # self.viewer.layers['input_image'].scale = vis_param['raw_rescale_factor']
            # self.viewer.layers['label_image'].scale = vis_param['label_rescale_factor']


            #apply a selected LUT to the label image - does not work, hence: add label as image layer above with additive blending
            # if vis_param['set_label_colormap'] != 'default':
            #     self.viewer.layers['label_image'].colormap = napari.utils.colormaps.colormap_utils.vispy_or_mpl_colormap(vis_param['set_label_colormap'])


            # save a first camera position
            self.viewer.camera.angles = vis_param['camera_angle1']
            imagereturn = self.viewer.screenshot(canvas_only=True, scale=vis_param['scale_to_save'])
            imwrite(visualized_file, imagereturn)

            # if you want to save a second camera position
            # get angle from napari by entering: viewer.camera.angles in console
            self.viewer.camera.angles = vis_param['camera_angle2']
            imagereturn2 = self.viewer.screenshot(canvas_only=True,scale=vis_param['scale_to_save'])
            imwrite(visualized_file2, imagereturn2)

            # if you establish the parameters, run napari, otherwise delete the layers for next timepoint
            if vis_param['establish_param'] == 1:
                napari.run()
            else:
                self.viewer.layers.remove('input_image')
                self.viewer.layers.remove('label_image')
                self.viewer.layers.remove('input_image_vessels')
                self.viewer.layers.remove('label_image_red')
                self.viewer.layers.remove('label_image_green')
                self.viewer.layers.remove('label_image_blue')


if __name__ == '__main__':
    #vis_param = visualization_param
    visualization_param = dict(
        camera_angle1 = (169, -18, 72),
        camera_angle2 = (18, -53, -120),
        camera_zoom=0.7,
        raw_contrast_limits=(300, 1400),
        raw_contrast_limits_vessels=(600, 15027),
        raw_gamma=0.9,
        raw_gamma_vessel=0.5,
        label_gamma=0.51,
        rendering_dimension=3,
        label_blending='translucent',
        #raw_rescale_factor =[9.210526, 1, 1],
        #label_rescale_factor = [9.210526, 1, 1],
        raw_rescale_factor=[1, 1, 1],
        label_rescale_factor=[1, 1, 1],
        establish_param =0,
        set_label_colormap='default',
        scale_to_save=5
    )

    imagevisu = image_visualizer_stitched()
    rawdatafolder_parent = '/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/macrophage_control/20230602_Daetwyler_Xenograft/Experiment0013_stitched'
    imagevisu.rawdatafolder = os.path.join(rawdatafolder_parent, "fish3")
    # segmented data
    imagevisu.segmentationfolder = os.path.join(rawdatafolder_parent, "fish3_segmented", "1_CH488_000000")
    imagevisu.visualizedfolder = imagevisu.rawdatafolder + "_visualized"
    imagevisu.load_visualize_images(visualization_param)
