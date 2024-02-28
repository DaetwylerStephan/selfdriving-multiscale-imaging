import numpy as np
from tifffile import imread, imwrite
import os
import cv2
import copy
import sys
import threading

sys.path.append('C://Users/Colfax-202008/PycharmProjects/ContextDriven_MicroscopeControl/multiScale')
from auxiliary_code.constants import Image_parameters
from automated_microscopy.template_matching import automated_templateMatching
from automated_microscopy.image_deposit import images_InMemory_class

from pystackreg import StackReg
import pystackreg

from matplotlib import pyplot as plt
from skimage import transform, io, exposure

class drift_correction:
    def __init__(self,
                 lowres_PosList,
                 highres_PosList,
                 lowres_zspacing,
                 highres_zspacing,
                 highresShape_width,
                 highresShape_height,
                 timepoint,
                 imageRepoLists,
                 debugfilepath="path.txt"):
        """
        :param lowres_PosList: the list of position of the low resolution imaging
        :param highres_PosList: the list of position of the high resolution imaging
        :param lowres_zspacing: the plane spacing of the low resolution stacks
        :param highres_zspacing: the plane spacing of the high resolution stacks
        :param highresShape_width: width of image, convention: np.array(height, width) and cv2.function(width, height)
        :param highresShape_height: height of image (lowres max 5092 / highres max 2048)
        :param lowresShape_width: width of image, convention: np.array(height, width) and cv2.function(width, height)
        :param lowresShape_height: height of low res image (lowres max 5092 / highres max 2048)
        :param filepath: (optional) filepath to logging the drift correction
        """

        #init it
        self.lowres_positionList = lowres_PosList
        self.highres_positionList = highres_PosList
        self.scalingfactor = 11.11 / 4.25 *1000 #scalingfactor for how much one mm is in pixel on low res view
        self.scalingfactorLowToHighres = 11.11 / 55.55 * 6.5 / 4.25 #scalingfactor for high/low res camera views
        self.calibration_height = 130 #pixel difference height between center of camera field of low res and high res
        self.calibration_width = 159 #pixel difference width between center of camera field of low res and high res
        self.lowres_zspacing = lowres_zspacing
        self.highres_zspacing = highres_zspacing
        self.highres_width = highresShape_width
        self.highres_height = highresShape_height
        self.completed = np.zeros(2) # an array to indicate whether the drift correction was completed.
        self.ImageRepo = imageRepoLists
        self.increase_crop_size = 1.5 #in transmission image - increase crop size of image for better template matching
        self.currenttimepoint = timepoint

        #check for filepath - if provided, enter debug mode to save temporary images to debug folder
        if debugfilepath =="path.txt":
            self.debugmode = False
        else:
            try:
                os.makedirs(debugfilepath)
            except:
                pass
            self.debugmode = True
        self.logfolder = debugfilepath

        self.templatematching = automated_templateMatching()

        self.Lock = threading.Lock()
        self.Lock_2 = threading.Lock()


    def calculate_drift_highRes(self, xyview, xzview, yzview, previousimage, z_step, PosNumber):
        '''
        calculate drift based on high resolution images from previous timepoint, and update position list
        :param xyview:
        :param xzview:
        :param yzview:
        :param previousimage: file path to previous time point image
        :param z_step:
        :param PosNumber: unique ID of stack
        :return:
        '''

        #load timepoint
        isExist = os.path.exists(previousimage)
        if not isExist:
            print("Wanted to make drift correction to reference image that does not exist")
            return #return if file does not exist - e.g. for first timepoint

        ref = imread(previousimage)

        ref_xy = ref[0:xyview.shape[0], 0:xyview.shape[1]]
        ref_yz = ref[0:xyview.shape[0], xyview.shape[1]:]
        ref_xz = ref[xyview.shape[0]:, 0:xyview.shape[1]]

        assert ref_xy.shape == xyview.shape
        assert ref_yz.shape == yzview.shape
        assert ref_xz.shape == xzview.shape

        correctX1, correctY1 = self.register_image(ref_xy, xyview, 'translation')
        correctZ1, correctY2 = self.register_image(ref_yz, yzview, 'translation')
        correctX2, correctZ2 = self.register_image(ref_xz, xzview, 'translation')

        print(correctX1, correctX2, correctY1, correctY2, correctZ1, correctZ2)

        correctX_mm = (1/1000.) * Image_parameters.xy_pixelsize_highres_um * (correctX1 + correctX2)/2.
        correctY_mm = (1/1000.) * Image_parameters.xy_pixelsize_highres_um * (correctY1 + correctY2)/2.
        correctZ_mm = (1/1000.) * z_step * (correctZ1 + correctZ2)/2.

        correctionarray = [0, correctX_mm, correctY_mm, correctZ_mm, 0, 0]
        print(correctX_mm,correctY_mm, correctZ_mm)

        x = self.highres_positionList[self._find_Index_of_PosNumber(PosNumber)]
        y = np.array(correctionarray).astype(np.float)
        #print("y:" + str(y))
        newposition = x + y
        #print(newposition)
        print("position list: " + str(self.highres_positionList))
        self.highres_positionList[self._find_Index_of_PosNumber(PosNumber)] = newposition

        print("position list updated: " + str(self.highres_positionList[self._find_Index_of_PosNumber(PosNumber)]))


    def find_corresponsingHighResTiles(self, LowResPosNumber):
        '''
        find all high resolution stacks that are closest to a given low res view (of the same angle)
        :param LowResPosNumber: position of the low resolution view in the lowres position list.
        :return: list of all highres stacks PosNumbers which are assigned to low resolution stack.
        '''

        self.Lock.acquire()
        list_highresregions = []
        for highresline in range(len(self.highres_positionList)):
            posnumberhighres = self.highres_positionList[highresline][5]
            lowresnb_current = self.find_closestLowResTile(posnumberhighres, return_number=True)
            print(lowresnb_current)
            if lowresnb_current == LowResPosNumber:
                highresPosNb = self.highres_positionList[highresline][5] #get unique ID from highres position
                list_highresregions.append((highresPosNb))
        self.Lock.release()
        return list_highresregions

    def _find_Index_of_PosNumber(self, PosNumber):
        self.Lock_2.acquire()
        index = -1
        for iter in range(len(self.highres_positionList)):
            if self.highres_positionList[iter][5] == PosNumber:
                index = iter
        self.Lock_2.release()
        return index

    def find_closestLowResTile(self, PosNumber, return_number=False):
        '''
        find corresponding low resolution stack to selected high-res region (PosNumber) (of the same angle).
        :param PosNumber: position of the highres view in the highres position list with unique ID.
        :param return_number: if True, return number e.g. 1, if False return string for filename "low_stack000"
        :return: the corresponding file name of the low resolution stack which is closest to the high res stack.
        '''

        highrespoint = np.array(self.highres_positionList[self._find_Index_of_PosNumber(PosNumber)][1:4])
        angle = int(float(self.highres_positionList[self._find_Index_of_PosNumber(PosNumber)][4]))

        positioniter = -1
        positionnumber = -1
        dist = -1
        for lowresline in range(len(self.lowres_positionList)):
            positioniter = positioniter + 1

            # get current position from list

            angleLow = int(float(self.lowres_positionList[lowresline][4]))
            lowrespoint = np.array(self.lowres_positionList[lowresline][1:4])
            if angle==angleLow:
                dist_current = np.linalg.norm(highrespoint - lowrespoint)
                if dist == -1:
                    dist = dist_current
                    pos_label_line = "low_stack" + f'{positioniter:03}'
                    positionnumber = positioniter
                if dist > dist_current:
                    dist = dist_current
                    pos_label_line = "low_stack" + f'{positioniter:03}'
                    positionnumber = positioniter

        if return_number == False:
            return pos_label_line
        else:
            return positionnumber

    def calculate_Lateral_drift(self, PosNumberID, mode='fluorescence'):
        '''
        calculates lateral drift correction based on low res view.
        :param PosNumber: which entry of the highres list with ID=PosNumber are you trying to correct?
        :param mode: what drift correction are you running? e.g. on transmission image or fluorescence image
        :return: (rownumber, columnnumber, crop_height, crop_width) - the position of the newly found view in the max projection
        '''

        lateralId = 0
        UpDownID = 1
        PosNumber = copy.deepcopy(PosNumberID)

        #transmission is different as there is no 1-to-1 correspondance between lowres and highres view. Therefore, one needs to first establish
        #an image of the region to follow. This image is saved in the image list
        #if mode=="transmission":
        if 1==1:
            print('Use transmission image for drift correction')

            # retrieve corresponding high res image
            image_transmission = self.ImageRepo.image_retrieval("current_transmissionImage", PosNumber)

            #establish the position of the first tile.
            if image_transmission.shape[0] < 2:
                print('Establish first corresponding transmission image')

                #find corresponding low resolution tile to high resolution image and get coordinates.
                lowstacknumber = copy.deepcopy(self.find_closestLowResTile(PosNumber, return_number=True))
                coordinates_lowres = np.asarray(self.lowres_positionList[lowstacknumber][1:5])
                coordinates_highres = np.asarray(self.highres_positionList[self._find_Index_of_PosNumber(PosNumber)][1:5])

                # get low resolution image
                img_lowrestrans = self.ImageRepo.image_retrieval("current_lowRes_Proj", lowstacknumber)

                #calulate pixel coordinates from physical stage coordinates
                loc, pixel_width_highresInLowres, pixel_height_highresInLowres = self.calculate_pixelcoord_from_physicalcoordinates(coordinates_lowres, coordinates_highres, img_lowrestrans.shape)

                #retrieve region in lowres view of highres view
                corresponding_lowres_view = img_lowrestrans[loc[0]:(loc[0]+pixel_height_highresInLowres), loc[1]:(loc[1]+pixel_width_highresInLowres)]
                print("corr" + str(corresponding_lowres_view.shape))

                #assign the image to the image list
                self.ImageRepo.replaceImage("current_transmissionImage", PosNumber, copy.deepcopy(corresponding_lowres_view))

                # debug mode - save image
                if self.debugmode == True:
                    #generate filenames
                    file_maxproj = os.path.join(self.logfolder, "transmission_maxproj_" + str(PosNumber) + self.currenttimepoint + ".tif")
                    #save files
                    #img_rgb_sc = cv2.rectangle(img_lowrestrans, (loc[1], loc[0]), (loc[1] + pixel_w_highresInLowres, loc[0] + pixel_h_highresInLowres), (0, 255, 255), 2)
                    cv2.imwrite(file_maxproj, corresponding_lowres_view)

                return (loc[0], loc[1], pixel_height_highresInLowres, pixel_width_highresInLowres)
            else:
                # if not first time, take previous image and find corresponding image in translation image with template matching
                print("perform matching on transmission image")

                #get corresponding low res stack
                stacknumber = self.find_closestLowResTile(PosNumber, return_number=True)
                lowresstackimage = self.ImageRepo.image_retrieval("current_lowRes_Proj", stacknumber)
                currenthighresPosindex = copy.deepcopy(self._find_Index_of_PosNumber(PosNumber))

                #perform template matching
                (row_number, column_number) = self.templatematching.scaling_templateMatching_multiprocessing(lowresstackimage, image_transmission, 1)

                #update image in "current_transmissionImage" list
                pixel_width_highresInLowres = int(self.scalingfactorLowToHighres * self.highres_width * self.increase_crop_size)
                pixel_height_highresInLowres = int(self.scalingfactorLowToHighres * self.highres_height * self.increase_crop_size)
                current_crop_image = lowresstackimage[row_number:(row_number+pixel_height_highresInLowres), column_number:(column_number+pixel_width_highresInLowres)]
                self.ImageRepo.replaceImage("current_transmissionImage", PosNumber, copy.deepcopy(current_crop_image))


                #calculate the physical position of where to image highres stack
                #1.calculate middle position
                center_pixel = (int(lowresstackimage.shape[0]/2) + self.calibration_width, int(lowresstackimage.shape[1]/2) - self.calibration_height)

                print("center: " + str(center_pixel))
                print("pixel height: " + str(pixel_height_highresInLowres))
                print("pixel width" + str(pixel_width_highresInLowres))
                print("row " + str(row_number) + " column_number: " + str(column_number))
                print("scaling factor" + str(self.scalingfactor))
                print("lowrespos" + str(self.lowres_positionList[stacknumber]))
                row_number_middle = row_number + pixel_height_highresInLowres/2
                column_number_middle = column_number + pixel_width_highresInLowres/2
                mm_difference = ((row_number_middle - center_pixel[0])/self.scalingfactor, (column_number_middle - center_pixel[1])/self.scalingfactor)
                print("lateraldrift corr" + str(PosNumber) + ": " + str(mm_difference))

                #update position in highpositionlist
                #oldposition = self.highres_positionList[self._find_Index_of_PosNumber(PosNumber)]
                lowresposition = np.append(copy.deepcopy(self.lowres_positionList[stacknumber]), copy.deepcopy(PosNumber))
                correctionarray = np.array([0, -mm_difference[0], -mm_difference[1], 0, 0, 0]).astype(float)
                newposition = lowresposition + correctionarray
                newposition[0] = copy.deepcopy(int(self.highres_positionList[currenthighresPosindex][0]))
                axialposition = copy.deepcopy(self.highres_positionList[currenthighresPosindex][3]) #you don't want to overwrite the axial position

                print("before update: " + str(self.highres_positionList[currenthighresPosindex]))
                self.highres_positionList[currenthighresPosindex] = copy.deepcopy(newposition)
                self.highres_positionList[currenthighresPosindex][3] = axialposition
                print("after update: " + str(self.highres_positionList[currenthighresPosindex]))



                # debug mode - save image
                if self.debugmode == True:
                    # generate filenames
                    file_maxproj = os.path.join(self.logfolder, "transmission_maxproj_" + str(
                        int(PosNumber)) + self.currenttimepoint + ".tif")
                    # save files
                    cv2.imwrite(file_maxproj, current_crop_image)

                return (row_number, column_number, pixel_height_highresInLowres, pixel_width_highresInLowres)


    def calculate_pixelcoord_from_physicalcoordinates(self, coordinates_lowres, coordinates_highres, lowresshape):
        """
        calulate pixel coordinates from physical stage coordinates
        :param coordinates_lowres, stage coordinates of lowres image
        :param coordinates_highres, stage coordinates of highres image
        :return:
        """
        lateralId = 0
        UpDownID = 1

        # caclulate coordinate differences and scale them from mm to pixel values.
        coordinate_difference = coordinates_lowres - coordinates_highres
        coordinate_difference[lateralId] = coordinate_difference[lateralId] * self.scalingfactor
        coordinate_difference[UpDownID] = coordinate_difference[UpDownID] * self.scalingfactor

        print("coordinate diff" + str(coordinate_difference))

        # get central pixel for low resolution stack with calibration
        loc = (int(lowresshape[0] / 2 + self.calibration_width), int(lowresshape[1] / 2 - self.calibration_height) )
        print("loc " + str(loc))
        # highres size in lowres:
        pixel_width_highresInLowres = int(self.scalingfactorLowToHighres * self.highres_width * self.increase_crop_size)
        pixel_height_highresInLowres = int(self.scalingfactorLowToHighres * self.highres_height * self.increase_crop_size)



        # calculate displacement
        loc = (int(loc[0] + coordinate_difference[lateralId] - pixel_height_highresInLowres / 2),
               int(loc[1] + coordinate_difference[UpDownID] - pixel_width_highresInLowres / 2 ))

        print("first location" + str(loc))

        return loc, pixel_width_highresInLowres, pixel_height_highresInLowres

    def calculate_axialdrift(self, PosNumber, image1, image2, mode='fluorescence'):
        '''
        calculates axial drift correction based on low res view.
        :param image of axial max projection around laterally corrected position.
        :param PosNumber: which entry of the highres list are you trying to correct?
        :param mode: what drift correction are you running? e.g. on transmission image or fluorescence image
        :return:
        '''

        #if mode == "transmission":
        if 1==1:
            # retrieve corresponding high res image
            lowres_axial1_transmission = self.ImageRepo.image_retrieval("current_transmissionAxial1Image", PosNumber)

            # if return zero array, safe the image as first tile in the list.
            if lowres_axial1_transmission.shape[0] < 2:
                print("Establish axis images for transmission")
                self.ImageRepo.replaceImage("current_transmissionAxial1Image", PosNumber, copy.deepcopy(image1))
                self.ImageRepo.replaceImage("current_transmissionAxial2Image", PosNumber, copy.deepcopy(image2))

                # debug mode - save image
                if self.debugmode == True:
                    # generate filenames
                    file_axial1 = os.path.join(self.logfolder, "transmission_axial1proj_" + str(int(PosNumber)) + self.currenttimepoint + ".tif")
                    file_axial2 = os.path.join(self.logfolder, "transmission_axial2proj_" + str(int(PosNumber)) + self.currenttimepoint + ".tif")

                    # save files
                    cv2.imwrite(file_axial1, image1)
                    cv2.imwrite(file_axial2, image2)

            else:
                print("Calculate axial drift for transmission")

                # get corresponding low res stack
                correctX1, correctZ1 = self.register_image(lowres_axial1_transmission, image1, 'translation')
                print(correctX1, correctZ1)

                lowres_axial2_transmission = self.ImageRepo.image_retrieval("current_transmissionAxial2Image", PosNumber)
                correctY2, correctZ2 = self.register_image(lowres_axial2_transmission, image2, 'translation')

                print(correctY2, correctZ2)
                correctZ = (correctZ1 + correctZ2)/2

                #we are interested only in the

                # debug mode - save image
                if self.debugmode == True:
                    # generate filenames
                    file_axial1 = os.path.join(self.logfolder, "transmission_axial1proj_" + str(int(PosNumber)) + self.currenttimepoint + ".tif")
                    file_axial2 = os.path.join(self.logfolder, "transmission_axial2proj_" + str(int(PosNumber)) + self.currenttimepoint + ".tif")

                    # save files
                    cv2.imwrite(file_axial1, image1)
                    cv2.imwrite(file_axial2, image2)

                #divide by 1000000000 as these are the stage units in pm to mm conversion for the position list
                correctionFactor = self.lowres_zspacing/1000000000*correctZ

                #update position in highresposition list
                oldposition = self.highres_positionList[self._find_Index_of_PosNumber(PosNumber)]
                correctionarray = np.asarray([0, 0, 0, correctionFactor, 0, 0]).astype(float)
                newposition = oldposition + correctionarray
                self.highres_positionList[self._find_Index_of_PosNumber(PosNumber)] = newposition

                #Update image repository
                self.ImageRepo.replaceImage("current_transmissionAxial1Image", PosNumber, copy.deepcopy(image1))
                self.ImageRepo.replaceImage("current_transmissionAxial2Image", PosNumber, copy.deepcopy(image2))

                print("axial drift correction:" + str(correctionFactor))
                return (correctionFactor)

    def indicate_driftcorrectionCompleted(self, PosNumber):
        """
        :param PosNumber:
        :return: update completed array
        """
        index = self._find_Index_of_PosNumber(PosNumber)
        self.completed[index]=1

    def register_image(self, ref, mov, mode):
        """
        register two images (ref, mov) to each other using the StackReg
        :param ref: reference image
        :param mov: moving image
        :param mode: rigid (if mode=='rigid'), else translation
        :return: lateral (xshift, yshift)
        """
        if mode=='rigid':
            sr = StackReg(StackReg.RIGID_BODY)
        else:
            sr = StackReg(StackReg.TRANSLATION)
        reg = sr.register_transform(ref,mov)
        xshift = sr.get_matrix()[0,2]
        yshift = sr.get_matrix()[1,2]

        return xshift, yshift

    def plot_registration(self, ref, mov):
        '''
        plot images
        :param ref: image 1
        :param mov: image 2
        :return: plots them and their overlay
        '''

        f, ax = plt.subplots(3, 1, figsize=(18, 40))

        before_reg = self.composite_images([ref, mov])

        ax[0].imshow(ref, cmap='gray')
        ax[0].set_title('reference image')
        ax[0].axis('off')

        ax[1].imshow(mov, cmap='gray')
        ax[1].set_title('shifted image')
        ax[1].axis('off')

        ax[2].imshow(before_reg)
        ax[2].set_title('overlay')
        ax[2].axis('off');
        plt.show(block='False')

    def overlay_images(self, imgs, equalize=False, aggregator=np.mean):
        if equalize:
            imgs = [exposure.equalize_hist(img) for img in imgs]

        imgs = np.stack(imgs, axis=0)

        return aggregator(imgs, axis=0)

    def composite_images(self, imgs, equalize=False, aggregator=np.mean):
        if equalize:
            imgs = [exposure.equalize_hist(img) for img in imgs]

        imgs = [img / img.max() for img in imgs]

        if len(imgs) < 3:
            imgs += [np.zeros(shape=imgs[0].shape)] * (3 - len(imgs))

        imgs = np.dstack(imgs)

        return imgs





if __name__ == '__main__':

    #set test positions
    stage_PositionList = [ (1,1.13718, -24.69498, -1.0845, 0.0), (2, 0.5, 0.6, 0.7,4),(3, 0.5, 0.6, 0.7,0), (4, 0.5, 0.4, 0.7,0)]
    stage_highres_PositionList = [(1, 1.2441, -24.69498, -1.00431, 0.0, 1), (2, 1.2441, -25.25631, -0.89739, 0.0, 2)]

    #load test images
    img_lowres_t0000 = "D://test/drift_correctionTest/transmission_timeseries/lowres_transmission_t0000.tif"
    img_lowres_t0001 = "D://test/drift_correctionTest/transmission_timeseries/lowres_transmission100pixel_t0001.tif"
    img_lowres_t0002 = "D://test/drift_correctionTest/transmission_timeseries/lowres_transmission200pixel_t0002.tif"
    img_lowrestrans = imread(img_lowres_t0000)
    img_lowres_t_t0000 = imread(img_lowres_t0000)
    img_lowres_t_t0001 = imread(img_lowres_t0001)
    img_lowres_t_t0002 = imread(img_lowres_t0002)

    currenttime = "t00000"

    #initialize image repository class and drift correction class
    imagerepoclass = images_InMemory_class()
    c = drift_correction(stage_PositionList,
                         stage_highres_PositionList,
                         3.5 * 1000000,
                         0.3 * 1000000,
                         2048,
                         2048,
                         currenttime,
                         imagerepoclass,
                         debugfilepath="D://test/drift_correctionTest/transmission_timeseries_result")

    list = c.find_corresponsingHighResTiles(1)
    print(list)
    test = c._find_Index_of_PosNumber(1)
    print(test)

    image_first_axial1 = "D://test/drift_correctionTest/output1/axial1/transmission_axial1proj_1t00000.tif"
    image_first_axial2 = "D://test/drift_correctionTest/output1/axial2/transmission_axial2proj_1t00000.tif"
    img_first_axial1 = imread(image_first_axial1)
    img_first_axial2 = imread(image_first_axial2)


    image_old_axial1 = "D://test/drift_correctionTest/output1/axial1/transmission_axial1proj_1.0t00039.tif"
    image_new_axial1 = "D://test/drift_correctionTest/output1/axial1/transmission_axial1proj_1.0t00040.tif"
    image_old_axial2 = "D://test/drift_correctionTest/output1/axial2/transmission_axial2proj_1.0t00039.tif"
    image_new_axial2 = "D://test/drift_correctionTest/output1/axial2/transmission_axial2proj_1.0t00040.tif"
    img_old_axial1 = imread(image_old_axial1)
    img_new_axial1 = imread(image_new_axial1)
    img_old_axial2 = imread(image_old_axial2)
    img_new_axial2 = imread(image_new_axial2)


    c.ImageRepo.replaceImage("current_transmissionAxial1Image",1, img_first_axial1)
    c.ImageRepo.replaceImage("current_transmissionAxial2Image",1, img_first_axial2)
    c.calculate_axialdrift(1, img_new_axial1, img_new_axial2, mode='transmission')



    # ---------------------------------------------------------------------------------------------------
    #test: lateral drift

    #first acquisition
    c.ImageRepo.replaceImage("current_lowRes_Proj", 0, img_lowres_t_t0000)
    lat_drift1 = c.calculate_Lateral_drift(2, mode='transmission')
    print(lat_drift1)
    tmpimag1 = copy.deepcopy(c.ImageRepo.image_retrieval("current_transmissionImage", 2))

    #second acquisition
    c.currenttimepoint = "t00001"
    c.ImageRepo.replaceImage("current_lowRes_Proj", 0, img_lowres_t_t0001)
    #c.ImageRepo.replaceImage("current_lowRes_Proj", 0, img_lowres_t_t0000)
    lat_drift2 = c.calculate_Lateral_drift(2, mode='transmission')
    print(lat_drift2)
    tmpimag2 = copy.deepcopy(c.ImageRepo.image_retrieval("current_transmissionImage", 2))

    # # #third acquisition
    # c.currenttimepoint = "t00002"
    # c.ImageRepo.replaceImage("current_lowRes_Proj", 0, img_lowres_t_t0002)
    # lat_drift3 = c.calculate_Lateral_drift(2, mode='transmission')
    # print(lat_drift3)
    # tmpimag3 = copy.deepcopy(c.ImageRepo.image_retrieval("current_transmissionImage", 2))
    #
    # f2, ax2 = plt.subplots(3, 1, figsize=(18, 40))
    # ax2[0].imshow(tmpimag1, cmap='gray')
    # ax2[1].imshow(tmpimag2, cmap='gray')
    # ax2[2].imshow(tmpimag3, cmap='gray')
    # plt.show(block='False')

    #---------------------------------------------------------------------------------------------------
    #load sample images
    # img0name = "D://test/drift_correctionTest/CH488/t00000.tif"
    # img1name = "D://test/drift_correctionTest/CH488/t00001.tif"
    # # img0name ="D://multiScope_Data//20220421_Daetwyler_Xenograft//Experiment0007//projections//high_stack_001//CH488///t00000.tif"
    # # img1name ="D://multiScope_Data//20220421_Daetwyler_Xenograft//Experiment0007//projections//high_stack_001//CH488///t00001.tif"
    # #
    # img0 = imread(img0name)
    # img1 = imread(img1name)
    # img0_cropXY = img0[0:1024, 0:2048]
    # img1_cropXY = img1[0:1024, 0:2048]
    # img0_cropXZ = img0[1024:, 0:2048]
    # img1_cropXZ = img1[1024:, 0:2048]
    # img0_cropYZ = img0[0:1024, 2048:]
    # img1_cropZY = img1[0:1024, 2048:]
    #
    # img0_cropXY = img0[0:2048, 0:2048]
    # img1_cropXY = img1[0:2048, 0:2048]
    # img0_cropXZ = img0[2048:, 0:2048]
    # img1_cropXZ = img1[2048:, 0:2048]
    # img0_cropYZ = img0[0:2048, 2048:]
    # img1_cropZY = img1[0:2048, 2048:]
    #
    # c.plot_registration(img0_cropXY, img1_cropXY)
    # c.plot_registration(img0_cropXZ, img1_cropXZ)
    # c.plot_registration(img0_cropYZ, img1_cropZY)

    #c.calculate_drift_highRes(img1_cropXY, img1_cropXZ, img1_cropZY, "D://test/drift_correctionTest/CH488/t00000.tif", 0.3, 0)
    # c.calculate_drift_highRes(img1_cropXY, img1_cropXZ, img1_cropZY, "D://test/drift_correctionTest/CH488/t00000.tif", 0.3, 0)
    # c.calculate_drift_highRes(img1_cropXY, img1_cropXZ, img1_cropZY, "D://test/drift_correctionTest/CH488/t00000.tif", 0.3, 0)
    # c.calculate_drift_highRes(img1_cropXY, img1_cropXZ, img1_cropZY, "D://test/drift_correctionTest/CH488/t00000.tif", 0.3, 0)
    # #c.calculate_drift_highRes(img1_cropXY, img1_cropXZ, img1_cropZY, img0name, 0.3, 1)



    # pos = c.find_closestLowResTile(0)
    # print(pos)
    #
    # img_lowrestrans_name = "D://test/drift_correctionTest/transmission/lowres_transmission.tif"
    # img_lowres = imread(img_lowrestrans_name)
    # print(img_lowres.shape)
    # imagerepoclass.replaceImage("current_lowRes_Proj", 0, img_lowres)
    #
    # c.calculate_drift_lowRes_complete("D://test/drift_correctionTest/CH488/t00000.tif", 0, mode='transmission', firsttimepoint=True)
    #c.register_image(img0_crop,img1_crop,"translation")

