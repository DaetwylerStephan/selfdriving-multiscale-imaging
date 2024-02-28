import multiprocessing
import queue
import time
import os
import atexit
import threading
import numpy as np
from multiprocessing import shared_memory
from tifffile import imread, imwrite
import copy

import acquisition_array_class as acq_arrays

import src.camera.Photometrics_camera as Photometricscamera
import src.ni_board.vni as ni
import src.stages.rotation_stage_cmd as RotStage
import src.stages.translation_stage_cmd as TransStage
import src.filter_wheel.ludlcontrol as FilterWheel
import src.slit.slit_cmd as SlitControl
import src.voicecoil.voice_coil as Voice_Coil

import auxiliary_code.concurrency_tools as ct
import auxiliary_code.napari_in_subprocess as napari
from auxiliary_code.constants import FilterWheel_parameters
from auxiliary_code.constants import Stage_parameters
from auxiliary_code.constants import NI_board_parameters
from auxiliary_code.constants import Camera_parameters
from auxiliary_code.constants import ASLM_parameters

from automated_microscopy.drift_correction import drift_correction
from automated_microscopy.image_deposit import images_InMemory_class

class multiScopeModel:
    def __init__(
            self
    ):
        """
        The main model class of the multi-scale microscope.
        """
        self.unfinished_tasks = queue.Queue()

        self.num_frames = 0
        self.initial_time = time.perf_counter()

        # parameters
        self.exposure_time_HR = 200
        self.exposure_time_LR = 200
        self.continue_preview_lowres = False
        self.continue_preview_highres = False
        self.stack_nbplanes_lowres = 200
        self.stack_nbplanes_highres = 200
        self.lowres_planespacing = 10000000
        self.highres_planespacing = 10000000
        self.displayImStack = 1
        self.abortStackFlag = 0 #abort current stack acquisition if value is 1, otherwise 0

        # filepath variables for saving image and projections
        self.filepath = 'D:/acquisitions/testimage.tif'
        self.current_projectionfilepath = 'D:/acquisitions/testimage.tif'
        self.current_positionfilepath = 'D:/acquisitions/testimage.txt'
        self.experimentfilepath = 'D:/acquisitions'
        self.current_timepointstring = "t00000"
        self.past_timepointstring = "t00000"
        self.current_region = "low_stack001"

        self.current_laser = NI_board_parameters.laser488
        self.lowres_laserpower = [0, 0, 0, 0]
        self.highres_laserpower = [0, 0, 0, 0]

        self.channelIndicator = "00"
        self.slitopening_lowres = 3700
        self.slitopening_highres = 4558
        self.autoscale_preview = 0
        self.slow_velocity = 0.1
        self.slow_acceleration = 0.1

        self.current_lowresROI_width = Camera_parameters.LR_width_pixel
        self.current_lowresROI_height = Camera_parameters.LR_height_pixel
        self.current_highresROI_width = Camera_parameters.HR_width_pixel
        self.current_highresROI_height = Camera_parameters.HR_height_pixel

        # for keeping track of the allocated buffer size (so that it doesn't have to be allocated twice)
        self.updatebuffer_highres_width = 0
        self.updatebuffer_highres_height = 0
        self.updatebuffer_highres_stacknb = 0
        self.updatebuffer_lowres_width = 0
        self.updatebuffer_lowres_height = 0
        self.updatebuffer_lowres_stacknb = 0
        self.high_res_memory_names = None
        self.low_res_memory_names = None
        self.lowresbuffernumber = 3

        self.delay_cameratrigger = 0.004  # the time given for the stage to move to the new position

        self.ASLM_acquisition_time = 0.3
        self.ASLM_from_Volt = [0,0,0,0]  # first voltage applied at remote mirror
        self.ASLM_to_Volt = [0,0,0,0]  # voltage applied at remote mirror at the end of sawtooth
        self.ASLM_currentVolt = 0  # current voltage applied to remote mirror
        self.ASLM_staticLowResVolt = 0  # default ASLM low res voltage
        self.ASLM_staticHighResVolt = 0  # default ASLM high res voltage
        self.ASLM_alignmentOn = 0  # param=1 if ASLM alignment mode is on, otherwise zero
        self.ASLM_delaybeforevoltagereturn = 0.001  # 1 ms
        self.ASLM_additionalreturntime = 0.001  # 1ms
        self.ASLM_scanWidth = ASLM_parameters.simultaneous_lines

        # preview buffers
        self.low_res_buffer = np.zeros([Camera_parameters.LR_height_pixel, Camera_parameters.LR_width_pixel],
                                       dtype='uint16')
        self.high_res_buffer = ct.SharedNDArray(
            shape=(Camera_parameters.HR_height_pixel, Camera_parameters.HR_width_pixel), dtype='uint16')
        self.low_res_buffer.fill(0)  # fill to initialize
        self.high_res_buffer.fill(0)  # fill to initialize

        # textlabels for GUI
        self.currentFPS = str(0)

        # drift correction
        self.ImageRepo = images_InMemory_class() #place holder for obtaining drift correction class from controller
        self.driftcorrectionmodule = 0 #place holder for obtaining drift correction class from controller
        self.drift_correctionOnHighRes = 0 #parameter whether high res drift correction is enabled
        self.drift_correctionOnLowRes = 0 #parameter whether low res drift correction is enabled
        self.drift_which_channels = [0,0,0,0,0] #array on which channels drift correction is run
        self.perform_driftcorrectionOnChannel = 0 #flag whether for current stack, drift correction should be performed
        self.current_PosNumber = 0 #todo previously 'item0'

        # initialize buffers
        self.update_bufferSize()

        #initialize buffer queue so that there are two buffers - so that one can be used for acquisition and one for processing
        self.high_res_buffers_queue = queue.Queue()
        self.low_res_buffers_queue = queue.Queue()
        for i in range(2):
            self.high_res_buffers_queue.put(i)
        for i in range(self.lowresbuffernumber):
            self.low_res_buffers_queue.put(i)

        # start initializing all hardware component here by calling the initialization from a ResultThread
        lowres_camera_init = ct.ResultThread(target=self._init_lowres_camera).start()  # ~3.6s
        highres_camera_init = ct.ResultThread(target=self._init_highres_camera).start()  # ~3.6s
        self._init_display()  # ~1.3s

        # init acquisition array writing class
        self.get_acq_array = acq_arrays.acquisition_arrays(self)

        # initialize stages and stages in ResultThreads
        trans_stage_init = ct.ResultThread(target=self._init_XYZ_stage).start()  # ~0.4s
        rot_stage_init = ct.ResultThread(target=self._init_rotation_stage).start()
        filterwheel_init = ct.ResultThread(target=self._init_filterwheel).start()  # ~5.3s
        slit_init = ct.ResultThread(target=self._init_slit).start()  #

        self._init_ao()  # ~0.2s

        # wait for all started initialization threads before continuing (by calling thread join)
        lowres_camera_init.get_result()
        print('Successfully initialized lowres camera')
        highres_camera_init.get_result()
        print('Successfully initialized highres camera')
        filterwheel_init.get_result()
        print('Successfully initialized filter wheel')
        trans_stage_init.get_result()
        print('Successfully initialized stage')
        rot_stage_init.get_result()
        print('Successfully initialized rot stage')
        slit_init.get_result()
        print('Successfully initialized slit')


        print('Finished initializing multiScope')

#######################################################################################################################
# Next come the initialization functions for hardware, and smart microscopy tools
#######################################################################################################################
    def _init_lowres_camera(self):
        """
        Initialize low resolution camera
        """
        print("Initializing low resolution camera ..")
        # place the Photometrics class as object into an Object in Subprocess
        self.lowres_camera = ct.ObjectInSubprocess(Photometricscamera.Photo_Camera, 'PMPCIECam00')
        self.lowres_camera_ROI = self.lowres_camera.get_imageroi()
        print(self.lowres_camera_ROI)
        # self.lowres_camera.take_snapshot(20)
        print("done with camera.")

    def _init_highres_camera(self):
        """
        Initialize low resolution camera
        """
        print("Initializing high resolution camera..")
        # place the Photometrics class as object into an Object in Subprocess
        self.highres_camera = ct.ObjectInSubprocess(Photometricscamera.Photo_Camera, 'PMUSBCam00')
        self.highres_camera_ROI = self.highres_camera.get_imageroi()
        print(self.highres_camera_ROI)
        # self.lowres_camera.take_snapshot(20)
        print("done with camera.")

    def _init_voicecoil(self):
        """
        Initialize the voice coil
        :return: initialized voice coil
        """
        print("Initializing voice coil ..")
        self.voice_coil = Voice_Coil.VoiceCoil(verbose=True)
        self.voice_coil.send_command('k0\r')  # Turn off servo
        time.sleep(1)
        self.voice_coil.send_command('k1\r')  # Engage servo
        time.sleep(1)
        self.voice_coil.send_command('d\r')  # Engage servo

    def _init_display(self):
        print("Initializing display...")
        self.display = ct.ObjectInSubprocess(napari._NapariDisplay, custom_loop=napari._napari_child_loop,
                                             close_method_name='close')

        print("done with display.")

    def _init_ao(self):
        """
        Initialize National Instruments card 6378 as device 1, Dev1
        """
        print("Initializing ao card...", end=' ')

        self.ao = ni.Analog_Out(
            num_channels=NI_board_parameters.ao_nchannels,
            rate=NI_board_parameters.rate,
            daq_type=NI_board_parameters.ao_type,
            line=NI_board_parameters.line_selection,
            verbose=True)

        self.ao_laser488_power = ni.Analog_Out(
            daq_type=NI_board_parameters.ao_type_constant,
            line=NI_board_parameters.power_488_line,
            minVol=NI_board_parameters.minVol_constant,
            maxVol=NI_board_parameters.maxVol_constant,
            verbose=True)
        self.ao_laser552_power = ni.Analog_Out(
            daq_type=NI_board_parameters.ao_type_constant,
            line=NI_board_parameters.power_552_line,
            minVol=NI_board_parameters.minVol_constant,
            maxVol=NI_board_parameters.maxVol_constant,
            verbose=True)
        self.ao_laser594_power = ni.Analog_Out(
            daq_type=NI_board_parameters.ao_type_constant,
            line=NI_board_parameters.power_594_line,
            minVol=NI_board_parameters.minVol_constant,
            maxVol=NI_board_parameters.maxVol_constant,
            verbose=True)
        self.ao_laser640_power = ni.Analog_Out(
            daq_type=NI_board_parameters.ao_type_constant,
            line=NI_board_parameters.power_640_line,
            minVol=NI_board_parameters.minVol_constant,
            maxVol=NI_board_parameters.maxVol_constant,
            verbose=True)
        self.flipMirrorPosition_power = ni.Analog_Out(
            daq_type=NI_board_parameters.ao_type_constant,
            line=NI_board_parameters.flip_mirror_line,
            minVol=NI_board_parameters.minVol_constant,
            maxVol=NI_board_parameters.maxVol_constant,
            verbose=True)
        self.mSPIMmirror_voltage = ni.Analog_Out(
            daq_type=NI_board_parameters.ao_type_constant,
            line=NI_board_parameters.mSPIM_mirror_line,
            minVol=NI_board_parameters.minVol_constant,
            maxVol=NI_board_parameters.max_mSPIM_constant,
            verbose=True)
        self.LED_voltage = ni.Analog_Out(
            daq_type=NI_board_parameters.ao_type_constant,
            line=NI_board_parameters.LED_line,
            minVol=NI_board_parameters.minVol_constant,
            maxVol=NI_board_parameters.maxVol_constant,
            verbose=True)
        self.mSPIMmirror_voltage.setconstantvoltage(0.1)
        print("done with ao.")
        atexit.register(self.ao.close)

    def _init_filterwheel(self):
        """
        Initialize filterwheel
        """
        ComPort = FilterWheel_parameters.comport
        self.filters = FilterWheel_parameters.avail_filters

        print("Initializing filter wheel...", end=' ')
        self.filterwheel = FilterWheel.LudlFilterwheel(ComPort, self.filters)
        self.filterwheel.set_filter('515-30-25', wait_until_done=False)
        self.filterwheel.set_filter('572/20-25', wait_until_done=False)
        self.filterwheel.set_filter('615/20-25', wait_until_done=False)
        self.filterwheel.set_filter('676/37-25', wait_until_done=False)
        print("done with filterwheel.")

    def _init_XYZ_stage(self):
        """
        Initialize translation stage
        """
        print("Initializing XYZ stage usb:sn:MCS2-00001795...")
        stage_id = Stage_parameters.stage_id_XYZ
        self.XYZ_stage = TransStage.SLC_translationstage(stage_id)
        self.XYZ_stage.findReference()
        print("done with XYZ stage.")
        atexit.register(self.XYZ_stage.close)

    def _init_rotation_stage(self):
        """
        Initialize rotation stage
        """
        print("Initializing rotation stage...")
        stage_id = Stage_parameters.stage_id_rot
        self.rotationstage = RotStage.SR2812_rotationstage(stage_id)
        # self.rotationstage.ManualMove()
        print("done with rot stage.")
        atexit.register(self.rotationstage.close)

    def _init_slit(self):
        """
        Initialize motorized slit
        """
        self.adjustableslit = SlitControl.slit_ximc_control()
        self.adjustableslit.slit_info()
        self.adjustableslit.slit_status()
        self.adjustableslit.slit_set_microstep_mode_256()
        self.adjustableslit.home_stage()
        print("slit homed")
        self.adjustableslit.slit_set_speed(800)

#######################################################################################################################
# Next come the functions at the end of a microscopy session - closing functions
#######################################################################################################################
    def close(self):
        """
        Close all opened channels, camera etc
                """
        self.finish_all_tasks()
        self.lowres_camera.close()
        self.highres_camera.close()
        self.ao.close()
        self.rotationstage.close()
        self.XYZ_stage.close()
        self.adjustableslit.slit_closing()
        self.display.close()  # more work needed here
        print('Closed multiScope')

    def finish_all_tasks(self):
        collected_tasks = []
        while True:
            try:
                th = self.unfinished_tasks.get_nowait()
            except queue.Empty:
                break
            th.join()
            collected_tasks.append(th)
        return collected_tasks

#######################################################################################################################
# functions to control buffers and hardware run from GUI: update_bufferSize, set_laserpower, check_movementboundaries,
# move_to_position, move_adjustableslit, changeLRtoHR, changeHRtoLR
#######################################################################################################################

    def update_bufferSize(self):
        """
        This handles the size of the buffers during acquisitions.
        """

        # check for whether some image dimension parameters were changed
        if (self.updatebuffer_highres_width != self.current_highresROI_width) or (
                self.updatebuffer_highres_height != self.current_highresROI_height) or (
                self.updatebuffer_highres_stacknb != self.stack_nbplanes_highres):

            # make sure to delete previous shared memory
            if self.high_res_memory_names != None:
                print("Delete previous shared memory arrays")
                try:
                    shared_memory.SharedMemory(name=self.high_res_memory_names[0]).unlink()
                    shared_memory.SharedMemory(name=self.high_res_memory_names[1]).unlink()
                except FileNotFoundError:
                    pass  # This is the error we expected if the memory was unlinked.

            # allocate memory
            self.high_res_buffers = [ct.SharedNDArray(
                shape=(self.stack_nbplanes_highres, self.current_highresROI_height, self.current_highresROI_width),
                dtype='uint16')
                                     for i in range(2)]
            self.high_res_buffers[0].fill(0)
            self.high_res_buffers[1].fill(0)

            self.high_res_memory_names = [self.high_res_buffers[i].shared_memory.name for i in range(2)]
            # save current parameters so that you don't have to reallocate the memory again without image dimension changes
            self.updatebuffer_highres_width = self.current_highresROI_width
            self.updatebuffer_highres_height = self.current_highresROI_height
            self.updatebuffer_highres_stacknb = self.stack_nbplanes_highres
            print("high res buffer updated")

        if (self.updatebuffer_lowres_width != self.current_lowresROI_width) or (
                self.updatebuffer_lowres_height != self.current_lowresROI_height) or (
                self.updatebuffer_lowres_stacknb != self.stack_nbplanes_lowres):

            # make sure to delete previous shared memory
            if self.low_res_memory_names != None:
                print("Delete previous shared memory arrays")
                try:
                    for i in range(self.lowresbuffernumber):
                        shared_memory.SharedMemory(name=self.low_res_memory_names[i]).unlink()
                except FileNotFoundError:
                    pass  # This is the error we expected if the memory was unlinked.

            self.low_res_buffers = [
                ct.SharedNDArray(
                    shape=(self.stack_nbplanes_lowres, self.current_lowresROI_height, self.current_lowresROI_width),
                    dtype='uint16')
                for i in range(self.lowresbuffernumber)]

            for i in range(self.lowresbuffernumber):
                self.low_res_buffers[i].fill(0)

            self.updatebuffer_lowres_width = self.current_lowresROI_width
            self.updatebuffer_lowres_height = self.current_lowresROI_height
            self.updatebuffer_lowres_stacknb = self.stack_nbplanes_lowres
            print("low res buffer updated")
            self.low_res_memory_names = [self.low_res_buffers[i].shared_memory.name for i in range(self.lowresbuffernumber)]

    def set_laserpower(self, powersettings):
        self.ao_laser488_power.setconstantvoltage(powersettings[0])
        self.ao_laser552_power.setconstantvoltage(powersettings[1])
        self.ao_laser594_power.setconstantvoltage(powersettings[2])
        self.ao_laser640_power.setconstantvoltage(powersettings[3])

    def check_movementboundaries(self, array):
        '''
        :param array = [axialPosition, lateralPosition, updownPosition, anglePosition], a list of position the stages moves to
        :return: an array which has no out of range positions
        '''
        if array[0] > 20 * 1000000000:
            array[0] = 19.9 * 1000000000
        if array[0] < -20 * 1000000000:
            array[0] = -19.9 * 1000000000

        if array[1] > 20 * 1000000000:
            array[1] = 19.9 * 1000000000
        if array[1] < -20 * 1000000000:
            array[1] = -19.9 * 1000000000

        if array[2] > 41.9 * 1000000000:
            array[2] = 41.5 * 1000000000
        if array[2] < -41.9 * 1000000000:
            array[2] = -41.5 * 1000000000

        return array

    def move_to_position(self, positionlist):
        """
        move to specified position according to positionlist
        :param positionlist: list of positions in format, e.g. [44280000, -2000000000, -2587870000]
        :return:
        """
        positionlistInt = np.array(positionlist, dtype=np.int64)
        self.XYZ_stage.moveToPosition(positionlistInt[0:3])
        self.rotationstage.moveToAngle(positionlist[3])

    def move_adjustableslit(self, slitopening, wait=0):
        """
        :param slitopening: move to this slitopening;
        :param if wait==1 - wait for slit move to finish before continuing
        """
        self.adjustableslit.slit_move(int(slitopening), 0)
        if wait == 1:
            self.adjustableslit.slit_wait_for_stop(100)

    def changeLRtoHR(self):
        """
        change from low resolution to high resolution acquisition settings
        """
        self.flipMirrorPosition_power.setconstantvoltage(3)
        self.move_adjustableslit(self.slitopening_highres, 1)

    def changeHRtoLR(self):
        """
        change from high resolution to low resolution acquisition settings
        """
        self.flipMirrorPosition_power.setconstantvoltage(0)
        self.move_adjustableslit(self.slitopening_lowres, 1)

#######################################################################################################################
# Next come the run preview functions (low and highres preview)
#######################################################################################################################

    def preview_lowres(self):
        """
        starts a custody thread to run a low resolution preview.
        """

        def preview_lowres_task(custody):

            self.num_frames = 0
            self.initial_time = time.perf_counter()

            def laser_preview():
                while self.continue_preview_lowres:
                    basic_unit = self.get_acq_array.get_lowres_preview_array()
                    self.ao.set_verbose(verbosevalue=False)
                    self.ao.play_voltages(basic_unit, block=True)

            ct.ResultThread(target=laser_preview).start()

            currentlaserpower = -1
            currentexposuretime =-1

            while self.continue_preview_lowres:


                if self.exposure_time_LR != currentexposuretime:
                    self.lowres_camera.set_up_lowres_preview(self.exposure_time_LR)
                    currentexposuretime = self.exposure_time_LR

                if self.lowres_laserpower != currentlaserpower:
                    self.set_laserpower(self.lowres_laserpower)
                    currentlaserpower = self.lowres_laserpower

                custody.switch_from(None, to=self.lowres_camera)
                # self.lowres_camera.run_preview(out=self.low_res_buffer)
                self.lowres_camera.acquire_preview_tobuffer()
                self.low_res_buffer = self.lowres_camera.get_previewbuffer()
                # display
                custody.switch_from(self.lowres_camera, to=self.display)
                self.display.show_image_lowres(self.low_res_buffer)

                if self.autoscale_preview == 1:
                    minval = np.amin(self.low_res_buffer)
                    maxval = np.amax(self.low_res_buffer)
                    self.display.set_contrast(minval, maxval, "lowrespreview")
                    print("updated preview settings")

                custody.switch_from(self.display, to=None)
                self.num_frames += 1
                # calculate fps to display
                if self.num_frames == 100:
                    time_elapsed = time.perf_counter() - self.initial_time
                    print("%0.2f average FPS" % (self.num_frames / time_elapsed))
                    self.num_frames = 0
                    self.initial_time = time.perf_counter()

            self.lowres_camera.end_preview()

        # self.low_res_buffer = ct.SharedNDArray(shape=(self.current_lowresROI_height, self.current_lowresROI_width), dtype='uint16')
        self.low_res_buffer = np.zeros([self.current_lowresROI_height, self.current_lowresROI_width], dtype="uint16")
        self.lowres_camera.init_previewbuffer(self.current_lowresROI_height, self.current_lowresROI_width)
        self.continue_preview_lowres = True
        th = ct.CustodyThread(target=preview_lowres_task, first_resource=None)
        th.start()
        return th

    ###this is the code to run a high resolution preview with a static light-sheet
    def preview_highres_static(self):
        def preview_highres_task(custody):

            self.num_frames = 0
            self.initial_time = time.perf_counter()

            def laser_preview_highres():
                # old_laserline = 0
                while self.continue_preview_highres:
                    basic_unit = self.get_acq_array.get_highres_preview_array()
                    self.ao.play_voltages(basic_unit, block=True)

            # run laser as sub-thread that is terminated when the preview button is pressed (self.continue_preview_highres is false).
            ct.ResultThread(target=laser_preview_highres).start()

            while self.continue_preview_highres:
                self.set_laserpower(self.highres_laserpower)

                self.highres_camera.set_up_highrespreview(self.exposure_time_HR)
                self.num_frames += 1
                custody.switch_from(None, to=self.highres_camera)
                self.highres_camera.run_preview(out=self.high_res_buffer, flipimage=True)

                # display acquired image
                custody.switch_from(self.highres_camera, to=self.display)
                self.display.show_image_highres(self.high_res_buffer)

                if self.autoscale_preview == 1:
                    minval = np.amin(self.high_res_buffer)
                    maxval = np.amax(self.high_res_buffer)
                    self.display.set_contrast(minval, maxval, "highrespreview")

                custody.switch_from(self.display, to=None)

                if self.num_frames == 100:
                    time_elapsed = time.perf_counter() - self.initial_time
                    avg_FPS = (self.num_frames / time_elapsed)
                    print("%0.2f average FPS" % avg_FPS)
                    self.currentFPS = str(avg_FPS)
                    self.num_frames = 0
                    self.initial_time = time.perf_counter()

            self.highres_camera.end_preview()

        self.high_res_buffer = ct.SharedNDArray(shape=(self.current_highresROI_height, self.current_highresROI_width),
                                                dtype='uint16')
        self.continue_preview_highres = True
        th = ct.CustodyThread(target=preview_highres_task, first_resource=self.highres_camera)
        th.start()
        return th

    def calculate_ASLMparameters(self, desired_exposuretime):
        """
        calculate the parameters for an ASLM acquisition
        :param desired_exposuretime: the exposure time that is desired for the whole acquisition
        :return: set the important parameters for ASLM acquisitions
        """
        linedelay = Camera_parameters.highres_line_digitization_time
        nbrows = self.current_highresROI_height
        self.ASLM_lineExposure = int(np.ceil(desired_exposuretime / (1 + (1+nbrows) / self.ASLM_scanWidth)))
        self.ASLM_line_delay = int(np.ceil((desired_exposuretime - self.ASLM_lineExposure) / ((nbrows+1) * linedelay))) - 1
        self.ASLM_acquisition_time = (self.ASLM_line_delay + 1) * nbrows * linedelay + self.ASLM_lineExposure + (
                    self.ASLM_line_delay + 1) * linedelay

        print(
            "ASLM parameters are: {} exposure time, and {} line delay factor, {} total acquisition time for {} scan width".format(
                self.ASLM_lineExposure, self.ASLM_line_delay, self.ASLM_acquisition_time, self.ASLM_scanWidth))

    def preview_highres_ASLM(self):
        def preview_highresASLM_task(custody):

            self.num_frames = 0
            self.initial_time = time.perf_counter()

            while self.continue_preview_highres:
                # get/update latest laserpower
                self.set_laserpower(self.highres_laserpower)

                # calculate ALSM parameters
                self.calculate_ASLMparameters(self.exposure_time_HR)
                self.highres_camera.prepare_ASLM_acquisition(self.ASLM_lineExposure, self.ASLM_line_delay)

                # generate acquisition array
                basic_unit = self.get_acq_array.get_highresASLM_preview_array()
                print("array generated")

                custody.switch_from(None, to=self.highres_camera)

                # write voltages, indicate "False" so that the voltages are not set back to zero at the end (for the remote mirror)
                write_voltages_thread = ct.ResultThread(target=self.ao._write_voltages, args=(basic_unit, False),
                                                        ).start()

                # start camera thread to poll for new images
                def start_camera_streamASLMpreview():
                    self.highres_camera.run_preview_ASLM(out=self.high_res_buffer)

                camera_stream_thread_ASLMpreview = ct.ResultThread(target=start_camera_streamASLMpreview).start()

                # play voltages
                self.ao.play_voltages(block=True, force_final_zeros=False)

                print("voltages played")
                camera_stream_thread_ASLMpreview.get_result()
                print("camera thread returned")
                self.num_frames += 1

                # display
                custody.switch_from(self.highres_camera, to=self.display)
                self.display.show_image_highres(self.high_res_buffer)

                if self.autoscale_preview == 1:
                    minval = np.amin(self.high_res_buffer)
                    maxval = np.amax(self.high_res_buffer)
                    self.display.set_contrast(minval, maxval, "highrespreview")

                custody.switch_from(self.display, to=None)

                if self.num_frames == 100:
                    time_elapsed = time.perf_counter() - self.initial_time
                    print("%0.2f average FPS" % (self.num_frames / time_elapsed))
                    self.num_frames = 0
                    self.initial_time = time.perf_counter()

            # end preview by setting voltages back to zero
            end_unit = np.zeros((100, NI_board_parameters.ao_nchannels), np.dtype(np.float64))
            self.ao.play_voltages(voltages=end_unit, block=True, force_final_zeros=False)

            self.highres_camera.end_preview()

        self.high_res_buffer = ct.SharedNDArray(shape=(self.current_highresROI_height, self.current_highresROI_width),
                                                dtype='uint16')

        # parameters for preview
        self.continue_preview_highres = True

        # start preview custody thread
        th = ct.CustodyThread(target=preview_highresASLM_task, first_resource=self.highres_camera)
        th.start()
        return th

#######################################################################################################################
# below here are the stack acquisition functions
#######################################################################################################################

    def stack_acquisition_master(self, current_folder, current_position, whichlaser, resolutionmode):
        """
        Master to start stack acquisitions of different channels and resolution modes. Decides which stack acquisition method to call
        :param current_folder: folder to save the acquired data
        :param current_folder: folder to save the projected data
        :param current_startposition: start position for the stack streaming
        :param whichlaser: which channels to image
        :return:
        """
        #get current start position
        # get current position from list
        xpos = int(float(current_position[1]) * 1000000000)
        ypos = int(float(current_position[2]) * 1000000000)
        zpos = int(float(current_position[3]) * 1000000000)
        angle = int(float(current_position[4]) * 1000000)
        current_startposition = [zpos, xpos, ypos, angle]
        print(current_startposition)

        # first get the right laser power
        if resolutionmode == "low":
            self.set_laserpower(self.lowres_laserpower)
        if resolutionmode == "highASLM":
            self.set_laserpower(self.highres_laserpower)
        if resolutionmode == "highSPIM":
            self.set_laserpower(self.highres_laserpower)

        filename_image = ["1_CH488_000000.tif", "1_CH552_000000.tif", "1_CH594_000000.tif", "1_CH640_000000.tif", "1_CHLED_000000.tif"]
        channel_name = ["CH488", "CH552",  "CH594", "CH640",  "CHLED"]
        laser_param = [NI_board_parameters.laser488,
                       NI_board_parameters.laser552,
                       NI_board_parameters.laser594,
                       NI_board_parameters.laser640,
                       NI_board_parameters.led]

        # save current position to file for later reproducibility
        self.current_positionfilepath = os.path.join(self.experimentfilepath, "positions", self.current_region + ".txt")
        self.save_currentpositionToFile(self.current_positionfilepath, current_startposition)

        for w_i in range(len(whichlaser)):
            #if laser is selected do:
            if whichlaser[w_i] == 1 and (self.abortStackFlag == 0):
                print("acquire laser: " + channel_name[w_i])
                #generate filepaths for projections and drift correction
                current_filepath = os.path.join(current_folder, filename_image[w_i])

                self.current_projectionfilepath_three = os.path.join(self.experimentfilepath, "projections", "three", self.current_region, channel_name[w_i],
                                                  self.current_timepointstring + ".tif")
                self.current_projectionfilepath_XY = os.path.join(self.experimentfilepath, "projections", "XY",
                                                               self.current_region, channel_name[w_i],
                                                               self.current_timepointstring + ".tif")
                self.current_projectionfilepath_XZ = os.path.join(self.experimentfilepath, "projections", "XZ",
                                                               self.current_region, channel_name[w_i],
                                                               self.current_timepointstring + ".tif")
                self.current_projectionfilepath_YZ = os.path.join(self.experimentfilepath, "projections", "YZ",
                                                               self.current_region, channel_name[w_i],
                                                               self.current_timepointstring + ".tif")
                self.past_projectionfilepath = os.path.join(self.experimentfilepath, "projections", "three", self.current_region, channel_name[w_i],
                                                            self.past_timepointstring + ".tif")


                #set flag for drift correction on channel (high-res drift correction) - take info from GUI
                if self.drift_which_channels[w_i]==1:
                    self.perform_driftcorrectionOnChannel = 1
                else:
                    self.perform_driftcorrectionOnChannel = 0

                #select resolution level and start stack acquisition
                if resolutionmode == "low":
                    self.acquire_stack_lowres(current_startposition, laser_param[w_i], current_filepath)
                if resolutionmode == "highASLM":
                    print("acquire high res ALSM")
                    self.acquire_stack_highres(current_startposition, laser_param[w_i], current_filepath, "ASLM")
                if resolutionmode == "highSPIM":
                    print("acquire high res SPIM")
                    self.acquire_stack_highres(current_startposition, laser_param[w_i], current_filepath, "SPIM")

    def save_currentpositionToFile(self, filepath, current_startposition):
        '''
        saves current position to file/append it, called by stack_acquisition_master
        :param filepath:
        :return:
        '''
        # make positions folder if it does not exist to write file
        try:
            os.makedirs(os.path.dirname(filepath))
        except:
            print("folder not created")

        str_positionarray = str(current_startposition)

        # append current position to text file
        with open(filepath, 'a') as f:
            f.write(self.current_timepointstring)
            f.write(': ')
            f.writelines(str_positionarray)
            f.writelines('\n')

    def prepare_acquisition(self, current_startposition, laser):
        """
        prepare acquisition by moving filter wheel and stage system to the correct position
        """

        def movestage():
            self.move_to_position(current_startposition)

        thread_stagemove = ct.ResultThread(target=movestage).start()

        self.LED_voltage.setconstantvoltage(0)

        if laser == NI_board_parameters.laser488:
            self.filterwheel.set_filter('515-30-25', wait_until_done=False)
        if laser == NI_board_parameters.laser552:
            self.filterwheel.set_filter('572/20-25', wait_until_done=False)
        if laser == NI_board_parameters.laser594:
            self.filterwheel.set_filter('615/20-25', wait_until_done=False)
        if laser == NI_board_parameters.laser640:
            self.filterwheel.set_filter('676/37-25', wait_until_done=False)
        if laser == NI_board_parameters.led:
            self.filterwheel.set_filter('515-30-25', wait_until_done=False)
            self.LED_voltage.setconstantvoltage(4)

        self.update_bufferSize()

        thread_stagemove.get_result()

    def acquire_stack_lowres(self, current_startposition, current_laserline, filepath):
        def acquire_task(custody):
            print("new low res stack acquisition started...")
            custody.switch_from(None, to=self.lowres_camera)

            # prepare acquisition by moving filter wheel and stage, and set buffer size
            self.prepare_acquisition(current_startposition, current_laserline)

            # prepare camera for stack acquisition - put in thread so that program executes faster :)
            def prepare_camera():
                self.lowres_camera.prepare_stack_acquisition_seq(self.exposure_time_LR)

            camera_prepare_thread = ct.ResultThread(target=prepare_camera).start()

            # define NI board voltage array
            basic_unit = self.get_acq_array.get_lowRes_StackAq_array(current_laserline)
            control_array = np.tile(basic_unit, (
            self.stack_nbplanes_lowres + 1, 1))  # add +1 as you want to return to origin position

            # write voltages
            write_voltages_thread = ct.ResultThread(target=self.ao._write_voltages, args=(control_array,),
                                                    ).start()

            # set up stage
            self.XYZ_stage.streamStackAcquisition_externalTrigger_setup(self.stack_nbplanes_lowres,
                                                                        self.lowres_planespacing, self.slow_velocity,
                                                                        self.slow_acceleration)

            # wait for camera set up before proceeding
            camera_prepare_thread.get_result()

            # start thread on stage to wait for trigger
            def start_stage_stream():
                self.XYZ_stage.streamStackAcquisition_externalTrigger_waitEnd()

            stream_thread = ct.ResultThread(target=start_stage_stream).start()  # ~3.6s

            def start_camera_streamfast():
                self.lowres_camera.run_stack_acquisition_buffer_fast(self.stack_nbplanes_lowres,
                                                                     self.low_res_buffers[current_bufferiter])
                return

            start_camera_streamfast_thread = ct.ResultThread(target=start_camera_streamfast).start()

            # play voltages - you need to use "block true" as otherwise the program finishes without playing the voltages
            self.ao.play_voltages(block=True)

            stream_thread.get_result()
            start_camera_streamfast_thread.get_result()
            write_voltages_thread.get_result()

            custody.switch_from(self.lowres_camera, to=self.display)

            def saveimage():
                # save image
                try:
                    imwrite(filepath, self.low_res_buffers[current_bufferiter])
                except:
                    print("couldn't save image")

            savethread = ct.ResultThread(target=saveimage).start()

            if self.displayImStack == 1:
                self.display.show_stack(self.low_res_buffers[current_bufferiter])

            def calculate_projection_and_drift():
                # calculate projections
                filepathforprojection_three = copy.deepcopy(self.current_projectionfilepath_three)  # assign now as filepath is updated for next stack acquired
                filepathforprojection_XY = copy.deepcopy(self.current_projectionfilepath_XY)  # assign now as filepath is updated for next stack acquired
                filepathforprojection_XZ = copy.deepcopy(self.current_projectionfilepath_XZ)  # assign now as filepath is updated for next stack acquired
                filepathforprojection_YZ = copy.deepcopy(self.current_projectionfilepath_YZ)  # assign now as filepath is updated for next stack acquired
                posnumber_lowres = copy.deepcopy(self.current_PosNumber)
                bufferindex = copy.deepcopy(current_bufferiter)
                driftcorr_OnChannel = copy.deepcopy(self.perform_driftcorrectionOnChannel)

                t0 = time.perf_counter()
                maxproj_xy = np.max(self.low_res_buffers[bufferindex], axis=0)
                maxproj_xz = np.max(self.low_res_buffers[bufferindex], axis=1)
                maxproj_yz = np.max(self.low_res_buffers[bufferindex], axis=2)
                t1 = time.perf_counter() - t0
                print("max proj time: " + str(t1))

                ##display max projection
                all_proj = np.zeros([self.current_lowresROI_height + self.stack_nbplanes_lowres,
                                     self.current_lowresROI_width + self.stack_nbplanes_lowres], dtype="uint16")

                all_proj[0:self.current_lowresROI_height, 0:self.current_lowresROI_width] = maxproj_xy
                all_proj[self.current_lowresROI_height:, 0:self.current_lowresROI_width] = maxproj_xz
                all_proj[0:self.current_lowresROI_height, self.current_lowresROI_width:] = np.transpose(maxproj_yz)

                self.display.show_maxproj(all_proj)

                #save maxproj
                try:
                    os.makedirs(os.path.dirname(filepathforprojection_three))
                    os.makedirs(os.path.dirname(filepathforprojection_XY))
                    os.makedirs(os.path.dirname(filepathforprojection_XZ))
                    os.makedirs(os.path.dirname(filepathforprojection_YZ))
                except:
                    print("folder not created")

                try:
                    imwrite(filepathforprojection_three, all_proj)
                    imwrite(filepathforprojection_XY, maxproj_xy.astype("uint16"))
                    imwrite(filepathforprojection_XZ, maxproj_xz.astype("uint16"))
                    imwrite(filepathforprojection_YZ, np.transpose(maxproj_yz.astype("uint16")))
                except:
                    print("couldn't save projection image:" + filepathforprojection_three)

                #perform drift correction on low res images
                t0 = time.perf_counter()
                if driftcorr_OnChannel == 1:
                    print("perform drift correction...")
                    if self.drift_correctionOnLowRes == 1:
                        print("perform drift correction on LowRes...")

                        #set current parameters
                        #high&low position list are updated when stack acquisition is started in multiScale_main to not override previous calculation for other position
                        self.driftcorrectionmodule.currenttimepoint = self.current_timepointstring
                        self.driftcorrectionmodule.lowres_zspacing = self.lowres_planespacing
                        self.driftcorrectionmodule.highres_zspacing = self.highres_planespacing
                        self.driftcorrectionmodule.highres_height = self.current_highresROI_height
                        self.driftcorrectionmodule.highres_width = self.current_highresROI_width
                        self.driftcorrectionmodule.lowres_height = self.current_lowresROI_height
                        self.driftcorrectionmodule.lowres_width = self.current_lowresROI_width


                        self.driftcorrectionmodule.ImageRepo.replaceImage("current_lowRes_Proj", posnumber_lowres, maxproj_xy)
                        print("image replaced")
                        highreslistID = self.driftcorrectionmodule.find_corresponsingHighResTiles(posnumber_lowres)
                        print("highreslist to do drift correction on: " + str(highreslistID) + " of " + str(posnumber_lowres))
                        # add max projection to ImageRepo
                        if self.drift_transmission == 0:
                            currentmode = 'fluorescene'
                        else:
                            currentmode = 'transmission'

                        for iter in highreslistID:
                            print("---------------------------------------------------------------------------")
                            print("drift correction on " + str(iter) + " and posnumber lowres" + str(posnumber_lowres) + " and " + filepathforprojection_three)
                            (row_number, column_number, crop_height, crop_width) = self.driftcorrectionmodule.calculate_Lateral_drift(copy.deepcopy(iter), mode=currentmode)
                            image1 = np.max(self.low_res_buffers[bufferindex][:, row_number:row_number+crop_height, column_number:column_number+crop_width], axis=1)
                            image2 = np.max(self.low_res_buffers[bufferindex][:, row_number:row_number+crop_height, column_number:column_number+crop_width], axis=2)
                            self.driftcorrectionmodule.calculate_axialdrift(copy.deepcopy(iter), image1, image2, mode=currentmode)
                            self.driftcorrectionmodule.indicate_driftcorrectionCompleted(iter)


                t1 = time.perf_counter() - t0
                print("drift correction time: " + str(t1))

            projection_thread = ct.ResultThread(target=calculate_projection_and_drift).start()

            custody.switch_from(self.display, to=None)
            # savethread.get_result()
            # projection_thread.get_result()
            return

        ##navigate buffer queue - where to save current image: this allows you to acquire another stack, while the stack before is still being processed
        current_bufferiter = self.low_res_buffers_queue.get()  # get current buffer iter
        self.low_res_buffers_queue.put(current_bufferiter)  # add number to end of queue

        acquire_thread = ct.CustodyThread(target=acquire_task, first_resource=self.lowres_camera).start()
        acquire_thread.get_result()

    def acquire_stack_highres(self, current_startposition, current_laserline, filepath, modality):
        def acquire_taskHighResSPIM(custody):
            print("start")
            custody.switch_from(None, to=self.highres_camera)

            # prepare acquisition by moving filter wheel etc
            self.prepare_acquisition(current_startposition, current_laserline)

            if modality == "ASLM":
                # obtain ASLM parameters
                self.calculate_ASLMparameters(self.exposure_time_HR)

                # define NI board voltage array
                basic_unit = self.get_acq_array.get_highResASLM_StackAq_array(current_laserline)
                control_array = np.tile(basic_unit, (
                    self.stack_nbplanes_highres + 1, 1))  # add +1 as you want to return to origin position

                # smooth remote mirror voltage
                control_array[:, NI_board_parameters.voicecoil] = self.smooth_sawtooth(
                    control_array[:, NI_board_parameters.voicecoil],
                    window_len=self.ao.s2p(0.002))
                print("voltage array calculated")

                # prepare high res camera for stack acquisition
                self.highres_camera.prepare_ASLM_acquisition(self.ASLM_lineExposure, self.ASLM_line_delay)
            else:
                # define NI board voltage array
                basic_unit = self.get_acq_array.get_highResSPIM_StackAq_array(current_laserline)
                control_array = np.tile(basic_unit,
                                        (self.stack_nbplanes_highres + 1,
                                         1))  # add +1 as you want to return to origin position

                # prepare high res camera for stack acquisition
                self.highres_camera.prepare_stack_acquisition_highres(self.exposure_time_HR)
                print("camera initialized")

            # write voltages
            write_voltages_thread = ct.ResultThread(target=self.ao._write_voltages, args=(control_array,),
                                                    ).start()

            # set up stage
            self.XYZ_stage.streamStackAcquisition_externalTrigger_setup(self.stack_nbplanes_highres,
                                                                        self.highres_planespacing, self.slow_velocity,
                                                                        self.slow_acceleration)

            # start thread on stage to wait for trigger
            def start_stage_streamHighResSPIM():
                self.XYZ_stage.streamStackAcquisition_externalTrigger_waitEnd()

            stream_thread = ct.ResultThread(target=start_stage_streamHighResSPIM).start()  # ~3.6s

            def start_camera_streamHighResSPIM():
                self.highres_camera.run_stack_acquisition_buffer_fast(self.stack_nbplanes_highres,
                                                                      self.high_res_buffers[current_bufferiter], flipimage=True)
                return

            start_highrescamera_stream_thread = ct.ResultThread(target=start_camera_streamHighResSPIM).start()

            print("stage and camera threads waiting ...")

            # play voltages
            # you need to use "block true" as otherwise the program finishes without playing the voltages really
            self.ao.play_voltages(block=True)

            stream_thread.get_result()
            start_highrescamera_stream_thread.get_result()

            custody.switch_from(self.highres_camera, to=self.display)

            def saveimage_highresSPIM():
                # save image
                try:
                    imwrite(filepath, self.high_res_buffers[current_bufferiter])
                except:
                    print("couldn't save image")

            savethread = ct.ResultThread(target=saveimage_highresSPIM).start()


            def calculate_projection_highres():
                # calculate projections

                #todo - list all parameters here that change from stack to stack to avoid racing conditions....
                filepathforprojection_three = copy.deepcopy(self.current_projectionfilepath_three)  # assign now as filepath is updated for next stack acquired
                filepathforprojection_XY = copy.deepcopy(self.current_projectionfilepath_XY)  # assign now as filepath is updated for next stack acquired
                filepathforprojection_XZ = copy.deepcopy(self.current_projectionfilepath_XZ)  # assign now as filepath is updated for next stack acquired
                filepathforprojection_YZ = copy.deepcopy(self.current_projectionfilepath_YZ)  # assign now as filepath is updated for next stack acquired
                pastfilepathforprojection = copy.deepcopy(self.past_projectionfilepath)
                current_region_item = copy.deepcopy(self.current_PosNumber)
                bufferindex = copy.deepcopy(current_bufferiter)
                driftcorr_OnChannel = copy.deepcopy(self.perform_driftcorrectionOnChannel)


                t0 = time.perf_counter()
                maxproj_xy = np.max(self.high_res_buffers[bufferindex], axis=0)
                maxproj_xz = np.max(self.high_res_buffers[bufferindex], axis=1)
                maxproj_yz = np.max(self.high_res_buffers[bufferindex], axis=2)
                t1 = time.perf_counter() - t0

                print("time: " + str(t1))

                ##display max projection
                all_proj = np.zeros([self.current_highresROI_height + self.stack_nbplanes_highres,
                                     self.current_highresROI_width + self.stack_nbplanes_highres], dtype="uint16")
                all_proj[0:self.current_highresROI_height, 0:self.current_highresROI_width] = maxproj_xy
                all_proj[self.current_highresROI_height:, 0:self.current_highresROI_width] = maxproj_xz
                maxproj_yzTransposed = np.transpose(maxproj_yz)
                all_proj[0:self.current_highresROI_height, self.current_highresROI_width:] = maxproj_yzTransposed

                self.display.show_maxproj(all_proj)

                try:
                    os.makedirs(os.path.dirname(filepathforprojection_three))
                    os.makedirs(os.path.dirname(filepathforprojection_XY))
                    os.makedirs(os.path.dirname(filepathforprojection_XZ))
                    os.makedirs(os.path.dirname(filepathforprojection_YZ))
                except:
                    print("folder not created")

                try:
                    imwrite(filepathforprojection_three, all_proj)
                    imwrite(filepathforprojection_XY, maxproj_xy.astype("uint16"))
                    imwrite(filepathforprojection_XZ, maxproj_xz.astype("uint16"))
                    imwrite(filepathforprojection_YZ, np.transpose(maxproj_yz.astype("uint16")))
                except:
                    print("couldn't save projection image")

                if self.displayImStack == 1:
                    self.display.show_stack(self.high_res_buffers[current_bufferiter])

                ###drift-correction module here for high res driftcorrection
                if driftcorr_OnChannel == 1:
                    if self.drift_correctionOnHighRes ==1:
                        current_region_iter = self.current_region.split('stack')[1]
                        self.driftcorrectionmodule.calculate_drift_highRes(maxproj_xy,
                                                                           maxproj_xz,
                                                                           maxproj_yzTransposed,
                                                                           pastfilepathforprojection,
                                                                           (self.highres_planespacing/1000000),
                                                                           current_region_item
                                                                           )


            projection_thread2 = ct.ResultThread(target=calculate_projection_highres).start()

            custody.switch_from(self.display, to=None)
            # savethread.get_result()

        ##navigate buffer queue - where to save current image.
        current_bufferiter = self.high_res_buffers_queue.get()  # get current buffer iter
        self.high_res_buffers_queue.put(current_bufferiter)  # add number to end of queue

        # start thread and wait for its completion
        acquire_threadHighResSPIM = ct.CustodyThread(
            target=acquire_taskHighResSPIM, first_resource=self.highres_camera).start()
        acquire_threadHighResSPIM.get_result()

    def smooth_sawtooth(self, array, window_len=101):

        if (window_len % 2) == 0:
            window_len = window_len + 1
        startwindow = int((window_len - 1) / 2)

        startarray = np.ones(startwindow) * array[0]
        endarray = np.ones(startwindow) * array[-1]

        s = np.r_[startarray, array, endarray]  # make array bigger on both sides

        w = np.ones(window_len, 'd')  # define a flat window - all values have equal weight

        returnarray = np.convolve(w / w.sum(), s, mode='valid')  # convolve with window to smooth

        return returnarray


if __name__ == '__main__':
    # first code to run in the multiscope

    # Create scope object:
    scope = multiScopeModel()
    # close
    scope.close()
