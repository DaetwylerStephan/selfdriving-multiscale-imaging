# Python program to illustrate
# multiscaling in template matching
import cv2
import numpy as np
import imutils
import copy
import time
import multiprocessing as mp
from itertools import repeat

from random import random


class automated_templateMatching:
    def __init__(self):
        self.Lock = mp.Lock()


    def simple_templateMatching(self, searchimage, template, scaling_factor, showimage=False):
        '''

        :param template: highres image which we want to find in the low-res / big image
        :param searchimage: the big image, in which we want to find the template
        :param scaling_factor: scaling of the highres image (template) to match image dimensions of low res image
        :param showimage: show image when executing template matching
        :return:(row_number, column_number) of the max value
        '''

        #convert image to unit32 file type for OpenCV template matching algorithm
        template.astype(np.uint32)
        searchimage.astype(np.uint32)

        # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        #resize highres image to fit dimensions of low res image - in our case
        template_resized = imutils.resize(template, width=int(template.shape[1] * scaling_factor))
        (tH, tW) = template_resized.shape[:2]

        #initiate template matching
        found = None

        # Perform match operations with normalized correlation
        res = cv2.matchTemplate(searchimage, template_resized, cv2.TM_CCOEFF_NORMED)

        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
        print(maxLoc)
        print(maxVal)

        # # Specify a threshold
        # threshold = 0.53035
        # # Store the coordinates of matched area in a numpy array
        # loc = np.where(res >= threshold)
        #
        # if showimage==True:
        #     # Draw a rectangle around the matched region.
        #     for pt in zip(*loc[::-1]):
        #         img_rgb = cv2.rectangle(searchimage, pt, (pt[0] + tW, pt[1] + tH), (0, 255, 255), 2)
        #
        #     # Show the final image with the matched area.
        #     img_rgb = cv2.resize(img_rgb, (1011, 592))
        #     cv2.imshow('Detected', img_rgb)
        #     cv2.waitKey(0)
        #
        #     cv2.imwrite('D://test/test_templatematching/template_result3.tif', img_rgb)

        loc = found[1]
        print("new location: " + str(loc))
        if showimage == True:
            # Draw a rectangle around the matched region.
            print(tH, tW)
            img_rgb_sc = cv2.rectangle(searchimage, loc, (loc[0] + tW, loc[1] + tH), (0, 255, 255), 2)

            # Show the final image with the matched area.
            img_rgb_sc = cv2.resize(searchimage, (1011, 592))
            cv2.imshow('Detected', img_rgb_sc)
            cv2.waitKey(0)

        # cv2 has a different numbering than np array
        row_number = loc[1]
        column_number = loc[0]
        return (row_number, column_number)


    def scaling_templateMatching(self, searchimage_sc, template_sc, scaling_factor, showimage=False):
        '''
        :param template: highres image which we want to find in the low-res / big image
        :param searchimage: the big image, in which we want to find the template
        :param scaling_factor: scaling of the highres image (template) to match image dimensions of low res image
        :param showimage: show image when executing template matching
        :return:
        '''

        #convert image to unit32 file type for OpenCV template matching algorithm
        template_sc.astype(np.uint32)
        searchimage_sc.astype(np.uint32)

        # template_sc = cv2.cvtColor(template_sc, cv2.COLOR_BGR2GRAY)
        # searchimage_sc = cv2.cvtColor(searchimage_sc, cv2.COLOR_BGR2GRAY)

        #resize highres image to fit dimensions of low res image - in our case
        template_resized = imutils.resize(template_sc, width=int(template_sc.shape[1] * scaling_factor))
        (tH, tW) = template_resized.shape[:2]

        #initiate template matching
        found = None

        #######
        maxvalue =0
        for scale in np.linspace(0.9, 1.1, 11):
            #resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(template_resized, width=int(template_resized.shape[1] * scale))
            r = resized.shape[1] / float(template_resized.shape[1])
            #print(r)
            # if the resized image is smaller than the template, then break
            # from the loop
            if searchimage_sc.shape[0] < tH or searchimage_sc.shape[1] < tW:
                break

            # matching to find the template in the image
            #edged = cv2.Canny(resized, 50, 200)
            searchimage_sc = searchimage_sc.astype("float32")
            resized = resized.astype("float32")

            res = cv2.matchTemplate(searchimage_sc, resized, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
            # if we have found a new maximum correlation value, then update
            # the found variable if found is None or maxVal > found[0]:
            if maxVal > maxvalue:
                found = (maxVal, maxLoc, r)
                maxvalue = maxVal
                #print("-----new value:------")
                #print(maxvalue)

        # Store the coordinates of matched area in a numpy array
        loc = found[1]
        print("new location: " + str(loc))
        if showimage==True:
            # Draw a rectangle around the matched region.
            print(tH, tW)
            img_rgb_sc = cv2.rectangle(searchimage_sc, loc, (loc[0] + tW, loc[1] + tH), (0, 255, 255), 2)

            # Show the final image with the matched area.
            img_rgb_sc = cv2.resize(searchimage_sc, (1011, 592))
            cv2.imshow('Detected', img_rgb_sc)
            cv2.waitKey(0)

        #cv2 has a different numbering than np array
        row_number = loc[1]
        column_number = loc[0]

        return (row_number, column_number)

    def scaling_templateMatching_multiprocessing(self, searchimage_sc, template_sc, scaling_factor, showimage=False):
        '''
        :param template: highres image which we want to find in the low-res / big image
        :param searchimage: the big image, in which we want to find the template
        :param scaling_factor: scaling of the highres image (template) to match image dimensions of low res image
        :param showimage: show image when executing template matching
        :return:
        '''

        #convert image to unit32 file type for OpenCV template matching algorithm
        template_sc.astype(np.uint32)
        searchimage_sc.astype(np.uint32)

        # template_sc = cv2.cvtColor(template_sc, cv2.COLOR_BGR2GRAY)
        # searchimage_sc = cv2.cvtColor(searchimage_sc, cv2.COLOR_BGR2GRAY)

        #resize highres image to fit dimensions of low res image - in our case
        template_resized = imutils.resize(template_sc, width=int(template_sc.shape[1] * scaling_factor))
        (tH, tW) = template_resized.shape[:2]

        #initiate template matching



        # create the shared lock
        lock = mp.Lock()
        queue = mp.Queue()
        # create a number of processes with different sleep times
        processes = [mp.Process(target=self.template_processing_subprocess, args=(lock,
                                                        scale,
                                                        template_resized,
                                                        searchimage_sc,
                                                        tH,
                                                        tW,
                                                        queue,)) for scale in np.linspace(0.9, 1.1, 11)]

        #processes = [mp.Process(target=self.task, args=(lock, i, random())) for i in range(10)]
        # start the processes
        for process in processes:
            process.start()
        # wait for all processes to finish
        for process in processes:
            process.join()

        found = None
        maxvalue = 0
        while not queue.empty():
            results = queue.get()
            if results[0]>maxvalue:
                maxvalue = results[0]
                found = results

        # Store the coordinates of matched area in a numpy array
        loc = found[1]
        print("new location: " + str(loc))
        if showimage==True:
            # Draw a rectangle around the matched region.
            print(tH, tW)
            img_rgb_sc = cv2.rectangle(searchimage_sc, loc, (loc[0] + tW, loc[1] + tH), (0, 255, 255), 2)
            # Show the final image with the matched area.
            img_rgb_sc = cv2.resize(searchimage_sc, (1011, 592))
            cv2.imshow('Detected', img_rgb_sc)
            cv2.waitKey(0)

        #cv2 has a different numbering than np array
        row_number = loc[1]
        column_number = loc[0]

        return (row_number, column_number)

    def template_processing_subprocess(self, lock, scale, template_resized, searchimage_sc, tH, tW, queue):
        resized = imutils.resize(template_resized, width=int(template_resized.shape[1] * scale))
        r = resized.shape[1] / float(template_resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if searchimage_sc.shape[0] < tH or searchimage_sc.shape[1] < tW:
            return

        # matching to find the template in the image
        # edged = cv2.Canny(resized, 50, 200)
        searchimage_sc = searchimage_sc.astype("float32")
        resized = resized.astype("float32")

        res = cv2.matchTemplate(searchimage_sc, resized, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
        # if we have found a new maximum correlation value, then update
        # the found variable if found is None or maxVal > found[0]:
        lock.acquire()
        found = (maxVal, maxLoc, r)
        queue.put(found)
        lock.release()

if __name__ == '__main__':

    # Create class object:
    template_matchClass = automated_templateMatching()

    # Load the template image
    ##template = cv2.imread('D://test/test_templatematching/template.tif')
    ##templateHighres = cv2.imread('D://test/test_templatematching/template3.tif')
    #templateHighres = cv2.imread('D://multiScope_Data/20220729_Daetwyler_Xenograft/Experiment0001/projections/XY/high_stack_002/CH552/t00000.tif')
    #templateHighres = cv2.imread('Z://Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/tracking_examples/20220727_Daetwyler_Nuclei/Experiment0005/projections/XY/high_stack_001/CH488/t00000.tif')
    templateHighres = cv2.imread('D://multiScope_Data/20220826_Daetwyler_Xenograft/Experiment0005/projections/XY/high_stack_001/CH594/t00000.tif')

    # Load the search image
    #img_gray = cv2.imread('D://test/test_templatematching/searchImage3.tif')
    #img_gray = cv2.imread('D://multiScope_Data/20220729_Daetwyler_Xenograft/Experiment0001/projections/XY/low_stack005/CH552/t00000.tif')
    #img_gray = cv2.imread('Z://Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/tracking_examples/20220727_Daetwyler_Nuclei/Experiment0005/projections/XY/low_stack000/CH488/t00000.tif')
    img_gray = cv2.imread('D://multiScope_Data/20220826_Daetwyler_Xenograft/Experiment0005/projections/XY/low_stack000/CH594/t00000.tif')

    scaling_factor = 11.11 / 55.55 * 6.5 / 4.25

    #template_matchClass.simple_templateMatching(copy.deepcopy(img_gray), copy.deepcopy(templateHighres), scaling_factor, showimage=True)
    # print("-------------scaled version--------------")
    # t0 = time.perf_counter()
    # template_matchClass.scaling_templateMatching(copy.deepcopy(img_gray), copy.deepcopy(templateHighres), scaling_factor, showimage=False)
    # t1 = time.perf_counter() - t0
    # print("time: " + str(t1))

    print("-------------mp scaled version--------------")
    t0 = time.perf_counter()
    template_matchClass.scaling_templateMatching_multiprocessing(copy.deepcopy(img_gray), copy.deepcopy(templateHighres),
                                                 scaling_factor, showimage=True)
    t1 = time.perf_counter() - t0
    print("time: " + str(t1))
