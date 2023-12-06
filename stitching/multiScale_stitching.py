# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from tifffile import imread, imwrite
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage as ndimage


class image_stitcher():
    """
    This class stitches images from the low-res multiscale
    """

    def __init__(self):
        """
        init stitcher
        """
        self.rawimages = []
        self.maxprojections = []
        self.expecteddisplacement_x = []
        self.expecteddisplacement_y = []
        self.calculated_displacement_x = []
        self.calculated_displacement_y = []
        self.correctionfactor = 1#make border regions a bit more bright if >1.
        self.reference_displacement_x =[]
        self.reference_displacement_y=[]

    def load_image(self, pathlist):
        """
        load the image specified by the path
        :param pathlist: array of images to open
        :return: populate the self.rawimages array
        """
        self.rawimages = []
        for imagepath_name in pathlist:
            self.rawimages.append(imread(imagepath_name))
            print("image loaded" + imagepath_name)

    def make_maxprojections(self):
        """
        make maximum projections of the images
        :return: populate the self.maxprojections array
        """
        self.maxprojections = []
        for image in self.rawimages:
            self.maxprojections.append(np.max(image,axis=0))

    def optimize_overlap(self, limit_search=[], extendfactor=100):
        """
        optimizes the provided overlap, based on the maximum projections
        :param limit_search: constrain the search around the cednter of a predefined position, e.g. what maximum displacement is expected
        :param extendfactor: depends on the overlapregion - how much you extend the search space and decrease it
        :return:
        """
        #check if maxprojections have been made
        if len(self.maxprojections)==0:
            self.make_maxprojections()

        translation_values_x = self.expecteddisplacement_x.copy()
        translation_values_y = self.expecteddisplacement_y.copy()

        for iter in range(len(self.maxprojections) - 1):
            imageshape = self.maxprojections[iter].shape
            print(str(imageshape))
            cut_y = translation_values_y[iter + 1] - translation_values_y[iter]
            # calculate overlap between image i and i+1

            # obtain overlap region image i
            overlaplength_y = imageshape[1] - cut_y

            image_first = self.maxprojections[iter].copy()
            image_second = self.maxprojections[iter + 1].copy()

            #determine relative x shift
            shift_x = translation_values_x[iter + 1] - translation_values_x[iter]
            if shift_x >0:
                start_image1_crop = int(shift_x)
                start_image2_crop = 0
                end_image1_crop = int(min(self.maxprojections[iter].shape[0], self.maxprojections[iter+1].shape[0] + shift_x))
                end_image2_crop = int(min(self.maxprojections[iter].shape[0] - shift_x, self.maxprojections[iter + 1].shape[0]))
            else:
                start_image1_crop = 0
                start_image2_crop = -int(shift_x)
                end_image1_crop = int(min(self.maxprojections[iter].shape[0],self.maxprojections[iter + 1].shape[0] + shift_x))
                end_image2_crop = int(min(-shift_x + self.maxprojections[iter].shape[0],self.maxprojections[iter + 1].shape[0]))

            print(start_image1_crop, start_image2_crop, end_image1_crop, end_image2_crop)

            crop_image_first = image_first[start_image1_crop:end_image1_crop, (cut_y - extendfactor):imageshape[1]].astype('float32')
            crop_image_second = image_second[start_image2_crop:end_image2_crop, 100:(overlaplength_y - extendfactor)].astype('float32')
            #imwrite(os.path.join(experimentfolder_result, "im_g1.tif"), crop_image_first)
            #imwrite(os.path.join(experimentfolder_result, "im_ga2.tif"), crop_image_second)

            im1_gaussian = np.subtract(crop_image_first, ndimage.gaussian_filter(crop_image_first, sigma=5))
            im2_gaussian = np.subtract(crop_image_second, ndimage.gaussian_filter(crop_image_second, sigma=5))

            # calculate the correlation image; note the flipping of one of the images
            fft_image = scipy.signal.fftconvolve(im1_gaussian, im2_gaussian[::-1, ::-1], mode='same')
            print(str(limit_search))
            if limit_search:
                c_rows= int(fft_image.shape[0]/2)
                c_columns= int(fft_image.shape[1]/2)
                print(c_rows)
                print(c_columns)
                fft_imagelimited = fft_image[c_rows - limit_search[0]:c_rows+limit_search[0], c_columns-limit_search[1]:c_columns+limit_search[1]]
                point = np.unravel_index(np.argmax(fft_imagelimited), fft_imagelimited.shape)

                print(point)
                corr_x = point[0] + (c_rows - limit_search[0])
                corr_y = point[1] + (c_columns-limit_search[1])
            else:
                point = np.unravel_index(np.argmax(fft_image), fft_image.shape)
                corr_x = point[0]
                corr_y = point[1]
            print(str(corr_x) + ", " + str(corr_y))
            print("currentiter " + str(iter) + ": " +  str((end_image1_crop - start_image1_crop)/ 2) + " and " + str(extendfactor + overlaplength_y / 2))
            # imwrite(os.path.join(experimentfolder_result, "im_gaussian1.tif"), im1_gaussian)

            # imwrite(os.path.join(experimentfolder_result, "im_gaussian2.tif"), im2_gaussian)
            correctionfactor_y = corr_y - extendfactor - overlaplength_y / 2
            translation_values_y[iter + 1] = translation_values_y[iter + 1] + int(correctionfactor_y)

            correctionfactor_x = int(corr_x - (end_image1_crop - start_image1_crop)/ 2)

            #compare x shift to previous shift to get an absolute displacement
            if iter==0:
                translation_values_x[iter + 1] = int(translation_values_x[iter+1]+ correctionfactor_x)
            else:
                translation_values_x[iter + 1] = int(translation_values_x[iter+1] + correctionfactor_x)

        ##normalize x shift, so that minimal value is zero
        translation_values_x = translation_values_x - np.min(translation_values_x)

        return (translation_values_x, translation_values_y)

    def perform_2D_maximum_fusion(self, imagearrays):
        """
        perform fusion by maximum selection of pixel
        :param imagearrays: list of images to fuse, translation based on self.calculated_displacement
        :return: fused image of imagearrays
        """
        imagelength_y = np.max(self.calculated_displacement_y) + imagearrays[len(imagearrays) - 1].shape[1]

        imagelength_x=0
        for i in range(len(self.maxprojections)):
            current_x = self.maxprojections[i].shape[0] + self.calculated_displacement_x[i]
            imagelength_x = max(current_x, imagelength_x)

        maximum_image = np.zeros(shape=(imagelength_x, imagelength_y), dtype='uint16')

        for iter in range(len(imagearrays)):
            imagetopad = imagearrays[iter].copy()
            startposition = self.calculated_displacement_y[iter]
            endposition = self.calculated_displacement_y[iter] + imagetopad.shape[1]
            startposition_x = self.calculated_displacement_x[iter]
            endposition_x = self.calculated_displacement_x[iter] + imagetopad.shape[0]

            paddedimage = np.pad(imagetopad, ((startposition_x, imagelength_x-endposition_x), (startposition, imagelength_y - endposition)), 'constant')
            maximum_image = np.maximum(maximum_image, paddedimage)

        return maximum_image

    def perform_2D_weighted_average_fusion(self, imagearrays):
        """
        perform fusion based on a sigmoidal blending of two stacks
        :param imagearrays: list of images to fuse
        :return: weighted average fused image
        """

        #calculate overall image shape
        imagelength_y = np.max(self.calculated_displacement_y) + imagearrays[len(imagearrays) - 1].shape[1]
        imagelength_x = 0
        for i in range(len(self.maxprojections)):
            current_x = self.maxprojections[i].shape[0] + self.calculated_displacement_x[i]
            imagelength_x = max(current_x, imagelength_x)

        ####calculate and apply weights
        weightedimages = imagearrays.copy()
        for iter in range(len(weightedimages) - 1):  # go over overlap regions

            # calculate overlap region parameters
            image = weightedimages[iter]
            nextimage = weightedimages[iter + 1]

            endposition_y = self.calculated_displacement_y[iter] + image.shape[1]
            startposition_next_y = self.calculated_displacement_y[iter + 1]
            overlapsize_y = endposition_y - startposition_next_y

            # define sigmoidal blending curves
            n1sigmoidrange = np.linspace(-6, 6, num=overlapsize_y) #use linspace not arrange here
            n1sigmoid = 1 / (1 + np.exp(n1sigmoidrange))
            inverse_1sigmoid = 1 - n1sigmoid

            #apply correction factor
            n1sigmoid = n1sigmoid*self.correctionfactor
            inverse_1sigmoid = inverse_1sigmoid*self.correctionfactor

            # make sigmoidal arrays
            n2sigmoid = np.tile(n1sigmoid, (image.shape[0], 1))
            n2_inversesigmoid = np.tile(inverse_1sigmoid, (nextimage.shape[0], 1))

            # make multiplication matrices
            invariableimagepart = np.ones(shape=(image.shape[0], image.shape[1] - overlapsize_y))
            imagemultiplicationarray = np.concatenate((invariableimagepart, n2sigmoid), axis=1)

            invariable_nextimage = np.ones(shape=(nextimage.shape[0], nextimage.shape[1] - overlapsize_y))
            nextimagemultiplicationarray = np.concatenate((n2_inversesigmoid, invariable_nextimage), axis=1)

            weightedimages[iter] = weightedimages[iter] * imagemultiplicationarray
            weightedimages[iter + 1] = weightedimages[iter + 1] * nextimagemultiplicationarray

        #add images together
        image_weighted_float = np.zeros(shape=(imagelength_x, imagelength_y), dtype='uint16')
        for iter in range(len(weightedimages)):  # addimages
            startposition_x = self.calculated_displacement_x[iter]
            endposition_x = self.calculated_displacement_x[iter] + weightedimages[iter].shape[0]
            startposition_y = self.calculated_displacement_y[iter]
            endposition_y = self.calculated_displacement_y[iter] + weightedimages[iter].shape[1]

            paddedimage = np.pad(weightedimages[iter], ((startposition_x, imagelength_x-endposition_x), (startposition_y, imagelength_y - endposition_y)), 'constant')
            image_weighted_float = image_weighted_float + paddedimage.copy()

        image_weighted = image_weighted_float.astype('uint16')

        return image_weighted

    def perform_3D_fusion(self, mode="weighted"):
        """
        perform fusion on 3D stack
        :param mode: maximum or weighted
        :return:
        """
        # calculate overall image shape
        imagelength_y = np.max(self.calculated_displacement_y) + self.rawimages[len(self.rawimages) - 1].shape[2]
        imagelength_x = 0
        for i in range(len(self.maxprojections)):
            current_x = self.maxprojections[i].shape[0] + self.calculated_displacement_x[i]
            imagelength_x = max(current_x, imagelength_x)
        nb_planes = self.rawimages[0].shape[0]

        print(str(imagelength_x) + ", " + str(imagelength_y))

        fused_image = np.zeros(shape=(nb_planes, imagelength_x, imagelength_y), dtype='uint16')

        #stitch plane by plane
        for i_plane in range(nb_planes):
            imagearray = []

            for i_image in range(len(self.rawimages)):
                imagearray.append(self.rawimages[i_image][i_plane, :, :])

            if mode=="weighted":
                fused_image[i_plane, :, :] = self.perform_2D_weighted_average_fusion(imagearray)
            else:
                fused_image[i_plane, :, :] = self.perform_2D_maximum_fusion(imagearray)

        return fused_image

    def crop_images(self,image_crops_x, image_crops_y):
        """
        remove parts of the images
        :param image_crops_x: for every image specify cropping in x direction [startpoint, endpoint], e.g. [[396, 396 + 3300], [208, 208 + 3236], [200, 200 + 3052]]
        :param image_crops_y: for every image specify cropping in y direction [startpoint, endpoint], e.g. [[396, 396 + 3300], [208, 208 + 3236], [200, 200 + 3052]]
        :return:
        """
        for iter in range(len(self.rawimages)):
            self.rawimages[iter] = self.rawimages[iter][:,(image_crops_x[iter][0]):(image_crops_x[iter][1]),(image_crops_y[iter][0]):(image_crops_y[iter][1])]

    def stitching_workflow(self,
                           imagelist,
                           image_crops_x=None,
                           image_crops_y=None,
                           optimizeAlignment=1,
                           limit_search=[],
                           clipped=1,
                           use_reference=1):
        """
        process one set of filepaths, load images, (optionally) crop them, (optionally) improve alignment, fuse data and
        (optionally) clip data.
        :param imagelist: filepaths of images to stitch
        :param image_crops_x: (optional) crop images before stitching
        :param image_crops_y: (optional) crop images before stitching
        :param optimizeAlignment: (optional) run optimization of alignment for stack (1 = yes, 0 = no)
        :param limit_search: (optional) if you want to limit the search to a region around a pre-aligned setting
        :param clipped: (optional) clip stack so that no zero value pixels are in the image (1 = yes, 0 = no)
        :param use_reference: (optional) use a reference set at the beginning for stitching so that parameters don't drift outside expected values
        :return: fused image of filepaths
        """

        #load images and crop if desired
        self.load_image(imagelist)
        if image_crops_x is not None:
            self.crop_images(image_crops_x, image_crops_y)

        #make max projection for stitching
        self.make_maxprojections()

        #optimize alignments
        if optimizeAlignment==1:
            print("calculate optimized alignment")
            (self.calculated_displacement_x, self.calculated_displacement_y) = self.optimize_overlap(limit_search=limit_search)
            self.expecteddisplacement_x = self.calculated_displacement_x
            self.expecteddisplacement_y = self.calculated_displacement_y
        elif self.calculated_displacement_y==[]:
            print("load expected displacement as parameters for stitching")
            self.calculated_displacement_x = self.expecteddisplacement_x
            self.calculated_displacement_y = self.expecteddisplacement_y

        print("calculated displacement y " + str(self.calculated_displacement_y))
        print("calculated displacement x " + str(self.calculated_displacement_x))

        fused_image = self.perform_3D_fusion()

        #clip to part with only data (remove 0 pixels at border
        if clipped == 1:
            lengthlist = []
            for i in range(len(self.maxprojections)):
                lengthlist.append(self.maxprojections[i].shape[0] + self.calculated_displacement_x[i])
            crop_end = np.min(lengthlist)
            print("clipped" + str(crop_end))
            fused_image2 = fused_image[:, np.max(i_stitch.calculated_displacement_x):crop_end, :]
        else:
            print("not cropped")
            fused_image2 = fused_image

        #set shift back to reference values
        if use_reference==1:
            self.expecteddisplacement_x = self.reference_displacement_x.copy()
            self.expecteddisplacement_y = self.reference_displacement_y.copy()

        return fused_image2

    def iterate_overfolder(self,
                           experimentfolder_parent,
                           experimentfolder_result,
                           channelnames,
                           region,
                           sample,
                           image_crops_y,
                           image_crops_x,
                           displacement_x,
                           displacement_y,
                           optimize_alignment=1,
                           limit_search=[],
                           use_reference=0
                           ):
        """
        :param experimentfolder_parent: folder where all timepoints are in
        :param experimentfolder_result: folder to save the result of stitching
        :param channelnames: list of channels to stitch
        :param region: list of regions, e.g. "low_stack000"
        :param sample: which fish/sample are you stitching, e.g. "fish3"
        :param image_crops_y: cropping parameters so that empty regions/at border can be removed, e.g. [[396, 396 + 3336], [308, 308 + 3336], [212, 212 + 3052]]
        :param image_crops_x: cropping parameters so that empty regions/at border can be removed, e.g. [[200, 2200], [200, 2200], [200, 2200]]
        :param displacement_x: approximate displacement in x
        :param displacement_y: approximate displacement in y
        :param optimize_alignment: (optional) do you want to optimize the alignment per timepoint or apply the same parameters to whole timelapse
        :param limit_search: (optional) do you want to restrict the finding of optimal stitching to a small overlap window? if yes, give a range: [100, 15]
        :param use_reference: use a reference for individual timepoints
        :return:
        """
        textfilepath = os.path.join(experimentfolder_result, sample + "_textfile.txt")


        # set starting parameters
        self.expecteddisplacement_x = displacement_x
        self.expecteddisplacement_y = displacement_y

        # get all timepoints of folder
        dir_list = os.listdir(experimentfolder_parent)
        timepointlist = []
        for path in dir_list:
            if path.startswith('t'):
                timepointlist.append(path)
        timepointlist.sort()
        print(timepointlist)

        #iterate over timepointlist
        for i_time in timepointlist:

           #if i_time < 't00016':
           #     print("skip")
           #     continue

            experimentfolder_timepoint = os.path.join(experimentfolder_parent, i_time)

            for imagechannelname in channelnames:
                imagelist = [os.path.join(experimentfolder_timepoint, region[0], imagechannelname + ".tif"),
                             os.path.join(experimentfolder_timepoint, region[1], imagechannelname + ".tif"),
                             os.path.join(experimentfolder_timepoint, region[2], imagechannelname + ".tif")]

                #check if you want to optimize alignment at all, if yes, which channel you want to optimize
                # (e.g. cancer cells have almost no signal and any alignment there is faulty
                optimize_channelalignment = optimize_alignment
                if optimize_alignment==1:
                    #only align vascular channel / one channel so that other channels get same stitching parameters
                    if imagechannelname=="1_CH594_000000":
                        optimize_channelalignment =1
                    else:
                        optimize_channelalignment =0


                fused_image = self.stitching_workflow(imagelist,
                                                      image_crops_x,
                                                      image_crops_y,
                                                      optimizeAlignment=optimize_channelalignment,
                                                      limit_search=limit_search,
                                                      use_reference=use_reference)

                savefolder = os.path.join(experimentfolder_result, sample, i_time)

                if not os.path.exists(savefolder):
                    os.makedirs(savefolder)

                save_file3D = os.path.join(savefolder, imagechannelname + ".tif")
                imwrite(save_file3D, fused_image)

                #make max projection for result verification
                maximage = np.max(fused_image, axis=0)
                maximageside = np.max(fused_image, axis=1)
                savefolder_max = os.path.join(experimentfolder_result, sample + "_max", imagechannelname)
                savefolder_max_side = os.path.join(experimentfolder_result, sample + "_max_side", imagechannelname)

                if not os.path.exists(savefolder_max):
                    os.makedirs(savefolder_max)
                if not os.path.exists(savefolder_max_side):
                    os.makedirs(savefolder_max_side)

                save_file_max = os.path.join(savefolder_max, i_time + ".tif")
                imwrite(save_file_max, maximage)
                save_file_max_side = os.path.join(savefolder_max_side, i_time + ".tif")
                imwrite(save_file_max_side, maximageside)

                file1 = open(textfilepath, "a")
                file1.write(i_time + ": " + str(self.calculated_displacement_x) + " displacement y: " + str(self.calculated_displacement_y) + "\n")
                file1.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # =============================================================================
    # initialization of class
    # =============================================================================
    i_stitch = image_stitcher()

    # =============================================================================
    # Load the images and generate save folder
    # =============================================================================
    experimentfolder_parent = "/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/macrophage_control/20230602_Daetwyler_Xenograft/Experiment0013"
    experimentfolder_result = experimentfolder_parent + "_stitched_test"
    channelnames = ["1_CH594_000000", "1_CH488_000000"]

    # =============================================================================
    # run parameters for stitching on all timepoints
    # =============================================================================

    region = ["low_stack006", "low_stack007", "low_stack008"]
    sample = "fish3"
    i_stitch.reference_displacement_x = [0, 45, 76]
    i_stitch.reference_displacement_y = [0, 2655, 5540]
    i_stitch.expecteddisplacement_x = i_stitch.reference_displacement_x.copy()
    i_stitch.expecteddisplacement_y = i_stitch.reference_displacement_y.copy()
    image_crops_y = [[396, 396 + 3300], [208, 208 + 3236], [200, 200 + 3452]]
    image_crops_x = [[170, 2650], [170, 2650], [170, 2650]]
    i_stitch.iterate_overfolder(experimentfolder_parent,
                                experimentfolder_result,
                                channelnames,
                                region,
                                sample,
                                image_crops_y,
                                image_crops_x,
                                i_stitch.expecteddisplacement_x,
                                i_stitch.expecteddisplacement_y,
                                optimize_alignment=1,
                                limit_search=[200, 80],
                                use_reference=1
                                )







