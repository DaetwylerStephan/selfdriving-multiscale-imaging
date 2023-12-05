import numpy as np
from tifffile import imread, imwrite
from matplotlib import pyplot as plt
import copy

class images_InMemory_class:
    def __init__(self):
        """
        This class is containing the images for calculations of drift correction, template matching and other smart microscopy
        """
        self.currentTP_lowResMaxProjection = []
        self.previousTP_lowResMaxProjection = []
        self.currentTP_highResMaxprojection = []
        self.previousTP_highResMaxProjection = []
        self.currentTP_highResAxial1projection = []
        self.previousTP_highResAxial1Projection = []
        self.currentTP_highResAxial2projection = []
        self.previousTP_highResAxial2Projection = []
        self.current_transmissionImageList = []
        self.previous_transmissionImageList = []
        self.current_transmissionAxial1ImageList = []
        self.previous_transmissionAxial1ImageList = []
        self.current_transmissionAxial2ImageList = []
        self.previous_transmissionAxial2ImageList = []

    def reset(self):
        """
        resets class to default value (without any images saved).
        """
        self.__init__()


    def addNewImage(self, whichlist, PosNumber, image):
        """
        add an Image newly to a list.
        :param whichlist: which list to add image - options: "current_lowRes_Proj", "previous_lowresProj",
                          "current_highRes_Proj", "previous_highRes_Proj", "current_transmissionImage",
                          "previous_transmissionImage", "current_highRes_Axial1Proj", "previous_highRes_Axial1Proj",
                          "current_highRes_Axial2Proj", "previous_highRes_Axial2Proj", "current_transmissionAxial1Image",
                          "current_transmissionAxial2Image","previous_transmissionAxial1Image", "previous_transmissionAxial2Image"
        :param PosNumber: the corresponding position number
        :param image:  the image to add
        :return: updated list in class
        """
        if whichlist == "current_lowRes_Proj":
            self.currentTP_lowResMaxProjection.append((PosNumber, image))
        if whichlist == "previous_lowresProj":
            self.previousTP_lowResMaxProjection.append((PosNumber, image))
        if whichlist == "current_highRes_Proj":
            self.currentTP_highResMaxprojection.append((PosNumber, image))
        if whichlist == "previous_highRes_Proj":
            self.previousTP_highResMaxProjection.append((PosNumber, image))
        if whichlist == "current_highRes_Axial1Proj":
            self.currentTP_highResAxial1projection.append((PosNumber, image))
        if whichlist == "previous_highRes_Axial1Proj":
            self.previousTP_highResAxial1Projection.append((PosNumber, image))
        if whichlist == "current_highRes_Axial2Proj":
            self.currentTP_highResAxial2projection.append((PosNumber, image))
        if whichlist == "previous_highRes_Axial2Proj":
            self.previousTP_highResAxial2Projection.append((PosNumber, image))
        if whichlist == "current_transmissionImage":
            self.current_transmissionImageList.append((PosNumber, image))
        if whichlist == "previous_transmissionImage":
            self.previous_transmissionImageList.append((PosNumber, image))
        if whichlist == "current_transmissionAxial1Image":
            self.current_transmissionAxial1ImageList.append((PosNumber, image))
        if whichlist == "previous_transmissionAxial1Image":
            self.previous_transmissionAxial1ImageList.append((PosNumber, image))
        if whichlist == "current_transmissionAxial2Image":
            self.current_transmissionAxial2ImageList.append((PosNumber, image))
        if whichlist == "previous_transmissionAxial2Image":
            self.previous_transmissionAxial2ImageList.append((PosNumber, image))

    def replaceImage(self, whichlist, PosNumber, image):
        """
        replace an Image with index PosNumber in the list "whichlist" with image.
        :param whichlist: which list to add image - options: "current_lowRes_Proj", "previous_lowresProj",
                          "current_highRes_Proj", "previous_highRes_Proj", "current_transmissionImage",
                          "previous_transmissionImage", "current_highRes_Axial1Proj", "previous_highRes_Axial1Proj",
                          "current_highRes_Axial2Proj", "previous_highRes_Axial2Proj", "current_transmissionAxial1Image",
                          "current_transmissionAxial2Image","previous_transmissionAxial1Image", "previous_transmissionAxial2Image"
        :param PosNumber: the corresponding position number
        :param image:  the image to add
        :return: updated list in class
        """
        if whichlist == "current_lowRes_Proj":
            self._updatelist(self.currentTP_lowResMaxProjection,
                             self.previousTP_lowResMaxProjection,
                             PosNumber, image, "current_lowRes_Proj", "previous_lowresProj")
        if whichlist == "current_highRes_Proj":
            self._updatelist(self.currentTP_highResMaxprojection,
                             self.previousTP_highResMaxProjection,
                             PosNumber, image, "current_highRes_Proj", "previous_highRes_Proj")
        if whichlist == "current_highRes_Axial1Proj":
            self._updatelist(self.currentTP_highResAxial1projection,
                             self.previousTP_highResAxial1Projection,
                             PosNumber, image, "current_highRes_Axial1Proj", "previous_highRes_Axial1Proj")
        if whichlist == "current_highRes_Axial2Proj":
            self._updatelist(self.currentTP_highResAxial2projection,
                             self.previousTP_highResAxial2Projection,
                             PosNumber, image, "current_highRes_Axial2Proj", "previous_highRes_Axial2Proj")
        if whichlist == "current_transmissionImage":
            self._updatelist(self.current_transmissionImageList,
                             self.previous_transmissionImageList,
                             PosNumber, image, "current_transmissionImage", "previous_transmissionImage")
        if whichlist == "current_transmissionAxial1Image":
            self._updatelist(self.current_transmissionAxial1ImageList,
                             self.previous_transmissionAxial1ImageList,
                             PosNumber, image, "current_transmissionAxial1Image", "previous_transmissionAxial1Image")
        if whichlist == "current_transmissionAxial2Image":
            self._updatelist(self.current_transmissionAxial2ImageList,
                             self.previous_transmissionAxial2ImageList,
                             PosNumber, image, "current_transmissionAxial2Image", "previous_transmissionAxial2Image")
        # if whichlist == "previous_highRes_Proj":
        #     self.previousTP_highResMaxProjection.append((PosNumber, image))
        # if whichlist == "transmission_ImageList":
        #     self.driftcorrection_transmissionImageList.append((PosNumber, image))

    def _updatelist(self, imagelist, previousimagelist, PosNumber, image, strcurrentlist, strpreviouslist):
        """
        helper function for replace image.
        :param imagelist: list to change
        :param previousimagelist: list of previous timepoint to change
        :param PosNumber: entry number
        :param image: image to update
        :param strcurrentlist: string to current list
        :param strpreviouslist: string to previous list
        :return:
        """
        found_image = 0

        #find entry in current image list, and update image in it
        for iter in range(len(imagelist)):
            if imagelist[iter][0] == PosNumber:
                found_image = 1
                temporaryimage = copy.deepcopy(imagelist[iter][1]) #copy so that it is not overwritten
                imagelist[iter] = (PosNumber, np.copy(image))

                #update previous time point list
                found_previous_image =0
                for iter2 in range(len(previousimagelist)):
                    if previousimagelist[iter2][0]==PosNumber:
                        previousimagelist[iter2] = (PosNumber, temporaryimage)
                        found_previous_image=1
                if found_previous_image==0: #if not found, add new image
                    self.addNewImage(strpreviouslist, PosNumber, temporaryimage)
        if found_image == 0:
            #print("add new image" + str(image.shape) + ":" + strcurrentlist)
            self.addNewImage(strcurrentlist, PosNumber, image)

    def image_retrieval(self, whichlist, PosNumber):
        """
        get an image from a list
        :param whichlist: which list to retrieve image - options: "current_lowRes_Proj", "previous_lowresProj",
                          "current_highRes_Proj", "previous_highRes_Proj", "current_transmissionImage",
                          "previous_transmissionImage", "current_highRes_Axial1Proj", "previous_highRes_Axial1Proj",
                          "current_highRes_Axial2Proj", "previous_highRes_Axial2Proj", "current_transmissionAxial1Image",
                          "current_transmissionAxial2Image","previous_transmissionAxial1Image", "previous_transmissionAxial2Image"
        :param PosNumber: what is the Position Number (PosNumber) associated with the image
        :return: image, or if image is not found returns an array with value zero: np.array([0]) for easy checking
        """
        if whichlist == "current_lowRes_Proj":
            returnimage = self._image_retrievalSupport(self.currentTP_lowResMaxProjection, PosNumber)
        if whichlist == "previous_lowresProj":
            returnimage = self._image_retrievalSupport(self.previousTP_lowResMaxProjection, PosNumber)
        if whichlist == "current_highRes_Proj":
            returnimage = self._image_retrievalSupport(self.currentTP_highResMaxprojection, PosNumber)
        if whichlist == "previous_highRes_Proj":
            returnimage = self._image_retrievalSupport(self.previousTP_highResMaxProjection, PosNumber)
        if whichlist == "current_highRes_Axial1Proj":
            returnimage = self._image_retrievalSupport(self.currentTP_highResAxial1projection, PosNumber)
        if whichlist == "previous_highRes_Axial1Proj":
            returnimage = self._image_retrievalSupport(self.previousTP_highResAxial1Projection, PosNumber)
        if whichlist == "current_highRes_Axial2Proj":
            returnimage = self._image_retrievalSupport(self.currentTP_highResAxial2projection, PosNumber)
        if whichlist == "previous_highRes_Axial2Proj":
            returnimage = self._image_retrievalSupport(self.previousTP_highResAxial2Projection, PosNumber)
        if whichlist == "current_transmissionImage":
            returnimage = self._image_retrievalSupport(self.current_transmissionImageList, PosNumber)
        if whichlist == "previous_transmissionImage":
            returnimage = self._image_retrievalSupport(self.previous_transmissionImageList, PosNumber)
        if whichlist == "current_transmissionAxial1Image":
            returnimage = self._image_retrievalSupport(self.current_transmissionAxial1ImageList, PosNumber)
        if whichlist == "previous_transmissionAxial1Image":
            returnimage = self._image_retrievalSupport(self.previous_transmissionAxial1ImageList, PosNumber)
        if whichlist == "current_transmissionAxial2Image":
            returnimage = self._image_retrievalSupport(self.current_transmissionAxial2ImageList, PosNumber)
        if whichlist == "previous_transmissionAxial2Image":
            returnimage = self._image_retrievalSupport(self.previous_transmissionAxial2ImageList, PosNumber)
        return returnimage

    def _image_retrievalSupport(self, currentlist, PosNumber):
        """
        support function for function image_retrieval, searches a list for the right PosNumber element.
        :param currentlist: list to search for PosNumber
        :param PosNumber: Indicator number of the saved image
        :return: image saved at the specific location - or an array with element 0
        """
        found_image = 0
        for iter in range(len(currentlist)):
            if currentlist[iter][0] == PosNumber:
                returnimage_2 = currentlist[iter][1]
                found_image = 1
        if found_image == 0:
            returnimage_2 = np.array([0])

        return returnimage_2


if __name__ == '__main__':

    #load some images to assign and replace images.
    image_deposit = images_InMemory_class()
    img_lowrestrans_name = "D://test/drift_correctionTest/transmission/lowres_transmission.tif"
    img_lowrestrans = imread(img_lowrestrans_name)
    img_crop_name = "D://test/drift_correctionTest/transmission/lowres_transmission_ROI.tif"
    img_crop = imread(img_crop_name)
    img_3 = "D://test/drift_correctionTest/transmission/lowres_transmission_found.tif"
    img_3 = imread(img_3)

    image_deposit.replaceImage("current_transmissionImage", 0, img_lowrestrans)
    image_deposit.replaceImage("current_transmissionImage", 1, img_3)
    image_deposit.replaceImage("current_transmissionImage", 2, img_lowrestrans)

    im1 = image_deposit.image_retrieval("current_transmissionImage", 1)
    im2 = image_deposit.image_retrieval("previous_transmissionImage", 1)

    f, ax = plt.subplots(2, 1, figsize=(18, 40))
    ax[0].imshow(img_lowrestrans, cmap='gray')
    ax[1].imshow(im1, cmap='gray')
    plt.show(block='False')

    image_deposit.replaceImage("current_transmissionImage", 0, img_crop)
    image_deposit.replaceImage("current_transmissionImage", 1, img_crop)
    image_deposit.replaceImage("current_transmissionImage", 2, img_crop)

    im3 = image_deposit.image_retrieval("current_transmissionImage", 1)
    im4 = image_deposit.image_retrieval("previous_transmissionImage", 1)
    print("next3")

    f, ax = plt.subplots(2, 1, figsize=(18, 40))
    ax[0].imshow(im3, cmap='gray')
    ax[1].imshow(im4, cmap='gray')
    plt.show(block='False')

    image_deposit.replaceImage("current_transmissionImage", 0, img_lowrestrans)
    image_deposit.replaceImage("current_transmissionImage", 1, img_3)
    image_deposit.replaceImage("current_transmissionImage", 2, img_lowrestrans)
    im1 = image_deposit.image_retrieval("current_transmissionImage", 1)
    im2 = image_deposit.image_retrieval("previous_transmissionImage", 1)

    f, ax = plt.subplots(2, 1, figsize=(18, 40))
    ax[0].imshow(im1, cmap='gray')
    ax[1].imshow(im2, cmap='gray')
    plt.show(block='False')


