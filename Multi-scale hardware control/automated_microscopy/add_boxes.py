
import os
import cv2
import numpy as np

def readintextfile(pathname, row_posarray, col_posarray):
    iter = 0
    with open(pathname,'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            #print(line)
            #print(line.split())
            splitarray = line.split("[")
            splitarray2 = splitarray[1].split("]")
            splitarray3 = splitarray2[0].split(",")
            #print(splitarray3)
            zaxis, row, col, angle = (int(val) for val in splitarray3)
            col_posarray[iter] = (col/1000000000)
            row_posarray[iter] = (row/1000000000)

            iter = iter+1


    return row_posarray, col_posarray

def function_calculator(coordinates_highres, parentfilepath):
    lateralId = 0
    UpDownID = 1
    scalingfactor = 11.11 / 4.25 * 1000
    scalingfactorLowToHighres = 11.11 / 55.55 * 6.5 / 4.25
    highres_width = 2048

    #vasculature fish
    highres_height = 1024
    increase_crop_size = 1.5
    coordinates_lowres =np.asarray([2574380000/1000000000, -20838850000/1000000000, 2584240000/1000000000, 0])

    # nuclei
    highres_height = 2048
    increase_crop_size = 1.5
    #t00000: [4061800000, 6305010000, -13685610000, 56000000]
    coordinates_lowres = np.asarray([6305010000/1000000000, -13685610000/1000000000, 4061800000/1000000000, 0])


    #coordinates_highres = np.asarray([2393360000/1000000000, -20809760000/1000000000, 2779830000/1000000000, 0])
    #coordinates_lowres = np.asarray(self.lowres_positionList[lowstacknumber][1:5])
    #coordinates_highres = np.asarray(self.highres_positionList[self._find_Index_of_PosNumber(PosNumber)][1:5])

    # caclulate coordinate differences and scale them from mm to pixel values.
    coordinate_difference = coordinates_lowres - coordinates_highres
    coordinate_difference[lateralId] = coordinate_difference[lateralId] * scalingfactor
    coordinate_difference[UpDownID] = coordinate_difference[UpDownID] * scalingfactor

    print("coordinate diff" + str(coordinate_difference))

    # get low resolution image
    file = os.path.join(parentfilepath, "t00000.tif")
    img_lowrestrans = cv2.imread(file)

    # get central pixel for low resolution stack
    loc = (int(img_lowrestrans.shape[0] / 2), int(img_lowrestrans.shape[1] / 2))
    print(loc)
    # highres size in lowres:
    pixel_width_highresInLowres = int(scalingfactorLowToHighres * highres_width * increase_crop_size)
    pixel_height_highresInLowres = int(scalingfactorLowToHighres * highres_height * increase_crop_size)

    # calculate displacement
    loc = (int(loc[0] + coordinate_difference[lateralId] - pixel_height_highresInLowres / 2),
           int(loc[1] + coordinate_difference[UpDownID] - pixel_width_highresInLowres / 2))

    print("first location" + str(loc))
    # # retrieve region in lowres view of highres view
    # corresponding_lowres_view = img_lowrestrans[loc[0]:(loc[0] + pixel_height_highresInLowres),
    #                             loc[1]:(loc[1] + pixel_width_highresInLowres)]
    # print("corr" + str(corresponding_lowres_view.shape))
    #
    # cv2.imwrite("D://test/boxes/annotated_regions1/test.tif", corresponding_lowres_view)
    return loc, pixel_height_highresInLowres, pixel_width_highresInLowres


if __name__ == '__main__':

    ###parameters

    parentfilepath = "Z://Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/tracking_examples/20220727_Daetwyler_Nuclei/Experiment0005/projections/XY/low_stack000/CH488"
    savefilepath = "D://test/boxes/annotated_nuclei/"

    textfile = "Z://Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/tracking_examples/20220727_Daetwyler_Nuclei/Experiment0005//positions//high_stack_001.txt"

    timepoints = 700
    colposarray = np.zeros(timepoints)
    rowposarray = np.zeros(timepoints)

    row_posarray, col_posarray = readintextfile(textfile, rowposarray, colposarray)





    for i in range(3):
        print(i)
        numStr = str(i).zfill(5)
        current_timepointstring = "t" + numStr + ".tif"
        print(current_timepointstring)
        currentfilepath = os.path.join(parentfilepath, current_timepointstring)
        currentsavefilepath= os.path.join(savefilepath, current_timepointstring)
        print(currentsavefilepath)

        currentarray = np.asarray([rowposarray[i], col_posarray[i], 0, 0])
        loc, tH, tW = function_calculator(currentarray, parentfilepath)
        loc_current = (loc[1], loc[0])
        tH = int(tH/3*2)
        tW = int(tW/3*2)
        current_img = cv2.imread(currentfilepath)

        img_rgb_sc = cv2.rectangle(current_img, loc_current, (loc_current[0] + tW, loc_current[1] + tH), (0, 255, 255), 3)
        cv2.imwrite(currentsavefilepath, img_rgb_sc)
    #
    # # #register_stack(filepath, savefilepath, mode="translation")
    # #
    # # directory = r"Y:\bioinformatics\Danuser_lab\Fiolka\LabMembers\Stephan\multiscale_data\data_Dagan\20220211_Daetwyler_Xenograft_DOXneg\Experiment0006"
    # # iterate_throughdirectory(directory)
