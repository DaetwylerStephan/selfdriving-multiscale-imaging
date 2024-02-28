import tkinter as tk
from tkinter import ttk

import time
from threading import Thread
import numpy as np
import datetime as dt
import os
import glob
import copy

from gui.main_window import MultiScope_MainGui
from multiScope import multiScopeModel
import auxiliary_code.concurrency_tools as ct
import auxiliary_code.write_parameters as write_params
from auxiliary_code.constants import Camera_parameters
from auxiliary_code.constants import NI_board_parameters
from auxiliary_code.constants import FileSave_parameters
from auxiliary_code.constants import ASLM_parameters
from automated_microscopy.drift_correction import drift_correction
from automated_microscopy.image_deposit import images_InMemory_class

class MultiScale_Microscope_Controller():
    """
    This is the controller in an MVC-scheme for mediating the interaction between the View (GUI) and the model (multiScope.py).
    Use: https://www.python-course.eu/tkinter_events_binds.php
    """

    #todo: check that stage values - e.g. with plane spacing, plane number don't expand beyond the possible travel range
    #todo: move to selected region in high resolution.

    def __init__(self):
        self.root = tk.Tk()

        # Create scope object as model
        self.model = multiScopeModel()
        print("model initiated")

        #create the gui as view
        all_tabs_mainGUI = ttk.Notebook(self.root)
        self.view = MultiScope_MainGui(all_tabs_mainGUI, self.model)

        #init param file writer class
        self.paramwriter = write_params.write_Params(self.view)

        #init drift correction module
        self.ImageRepo = images_InMemory_class()
        self.model.ImageRepo = self.ImageRepo
        self.drift_correctionmodule = drift_correction(self.view.stagessettingstab.stage_PositionList,
                                                       self.view.stagessettingstab.stage_highres_PositionList,
                                                       self.model.lowres_planespacing,
                                                       self.model.highres_planespacing,
                                                       self.model.current_highresROI_height,
                                                       self.model.current_highresROI_width,
                                                       self.model.current_timepointstring,
                                                       self.ImageRepo,
                                                       debugfilepath="D://test/drift_correctionTest/transmission_microscopeoutput_result")
        self.model.driftcorrectionmodule = self.drift_correctionmodule

        #define here which buttons run which function in the multiScope model
        self.continuetimelapse = 1 #enable functionality to stop timelapse

        #######connect buttons / variables from GUI with functions here-----------------------------------
        # connect all the buttons that start a functionality like preview, stack acquisition, etc.
        # connect all the buttons that you want to dynamically change, e.g. during preview

        #preview buttons
        self.view.runtab.bt_preview_lowres.bind("<ButtonRelease>", self.run_lowrespreview)
        self.view.runtab.bt_preview_highres.bind("<ButtonRelease>", self.run_highrespreview)
        self.view.runtab.bt_preview_stop.bind("<Button>", self.run_stop_preview)
        self.view.runtab.bt_changeTo488.bind("<Button>", lambda event: self.changefilter(event, '488'))
        self.view.runtab.bt_changeTo552.bind("<Button>", lambda event: self.changefilter(event, '552'))
        self.view.runtab.bt_changeTo594.bind("<Button>", lambda event: self.changefilter(event, '594'))
        self.view.runtab.bt_changeTo640.bind("<Button>", lambda event: self.changefilter(event, '640'))
        self.view.runtab.bt_changeTo_block.bind("<Button>", lambda event: self.changefilter(event, 'None'))
        self.view.runtab.bt_changeTo_trans.bind("<Button>", lambda event: self.changefilter(event, 'LED'))
        self.view.runtab.preview_autoIntensity.trace_add("write", self.updatepreview)

        #stack run buttons
        self.view.runtab.stack_aq_bt_run_stack.bind("<Button>", self.acquire_stack)
        self.view.runtab.stack_aq_bt_abort_stack.bind("<Button>", self.abort_stack)
        self.view.runtab.stack_aq_numberOfPlanes_highres.trace_add("write", self.update_stack_aq_parameters)
        self.view.runtab.stack_aq_numberOfPlanes_lowres.trace_add("write", self.update_stack_aq_parameters)
        self.view.runtab.stack_aq_plane_spacing_lowres.trace_add("write", self.update_stack_aq_parameters)
        self.view.runtab.stack_aq_plane_spacing_highres.trace_add("write", self.update_stack_aq_parameters)

        #timelapse run buttons
        self.view.runtab.timelapse_aq_bt_run_timelapse.bind("<Button>", self.acquire_timelapse)
        self.view.runtab.timelapse_aq_bt_abort_timelapse.bind("<Button>", self.abort_timelapse)

        #laser settings sliders
        self.view.runtab.laser488_percentage_LR.trace_add("read", self.updateLowResLaserParameters)
        self.view.runtab.laser552_percentage_LR.trace_add("read", self.updateLowResLaserParameters)
        self.view.runtab.laser594_percentage_LR.trace_add("read", self.updateLowResLaserParameters)
        self.view.runtab.laser640_percentage_LR.trace_add("read", self.updateLowResLaserParameters)
        self.view.runtab.laser488_percentage_HR.trace_add("read", self.updateHighResLaserParameters)
        self.view.runtab.laser552_percentage_HR.trace_add("read", self.updateHighResLaserParameters)
        self.view.runtab.laser594_percentage_HR.trace_add("read", self.updateHighResLaserParameters)
        self.view.runtab.laser640_percentage_HR.trace_add("read", self.updateHighResLaserParameters)

        #camera settings
        self.view.runtab.cam_lowresExposure.trace_add("write", self.updateExposureParameters)
        self.view.runtab.cam_highresExposure.trace_add("write", self.updateExposureParameters)

        #roi settings
        self.view.runtab.roi_applybutton.bind("<Button>", self.changeROI)

        #stage settings tab
        self.view.stagessettingstab.stage_moveto_axial.trace_add("write", self.movestage)
        self.view.stagessettingstab.stage_moveto_lateral.trace_add("write", self.movestage)
        self.view.stagessettingstab.stage_moveto_updown.trace_add("write", self.movestage)
        self.view.stagessettingstab.stage_moveto_angle.trace_add("write", self.movestage)
        self.view.stagessettingstab.keyboard_input_on_bt.bind("<Button>", self.enable_keyboard_movement)
        self.view.stagessettingstab.keyboard_input_off_bt.bind("<Button>", self.disable_keyboard_movement)
        self.view.stagessettingstab.move_to_specificPosition_Button.bind("<Button>", self.movestageToPosition)

        #advanced settings tab
        self.view.advancedSettingstab.slit_currentsetting.trace_add("write", self.slit_opening_move)
        self.view.advancedSettingstab.slit_lowres.trace_add("write", self.updateSlitParameters)
        self.view.advancedSettingstab.slit_highres.trace_add("write", self.updateSlitParameters)

        self.view.advancedSettingstab.stack_aq_stage_velocity.trace_add("write", self.update_stack_aq_parameters)
        self.view.advancedSettingstab.stack_aq_stage_acceleration.trace_add("write", self.update_stack_aq_parameters)
        self.view.advancedSettingstab.stack_aq_camera_delay.trace_add("write", self.update_stack_aq_parameters)

        self.view.advancedSettingstab.ASLM_volt_current.trace_add("write", self.update_ASLMParameters)
        self.view.advancedSettingstab.ASLM_alignmentmodeOn.trace_add("write", self.update_ASLMParameters)
        self.view.advancedSettingstab.ASLM_SawToothOn.trace_add("write", self.update_ASLMParameters)
        self.view.advancedSettingstab.ASLM_constantVoltageOn.trace_add("write", self.update_ASLMParameters)
        self.view.advancedSettingstab.ASLM_volt_interval.trace_add("write", self.update_ASLMParameters)
        self.view.advancedSettingstab.ASLM_volt_middle488.trace_add("write", self.update_ASLMParameters)
        self.view.advancedSettingstab.ASLM_volt_middle552.trace_add("write", self.update_ASLMParameters)
        self.view.advancedSettingstab.ASLM_volt_middle594.trace_add("write", self.update_ASLMParameters)
        self.view.advancedSettingstab.ASLM_volt_middle640.trace_add("write", self.update_ASLMParameters)

        self.view.advancedSettingstab.ASLM_voltageDirection.trace_add("write", self.update_ASLMParameters)
        self.view.advancedSettingstab.adv_settings_mSPIMvoltage.trace_add("write", self.update_mSPIMvoltage)
        self.view.advancedSettingstab.ASLM_scanWidth.trace_add("write", self.update_ASLMParameters)

        # smart settings tab
        self.view.automatedMicroscopySettingstab.drift_correction_highres.trace_add("write", self.updateDriftCorrectionSettings)
        self.view.automatedMicroscopySettingstab.drift_correction_lowres.trace_add("write", self.updateDriftCorrectionSettings)
        self.view.automatedMicroscopySettingstab.driftcorrection_488.trace_add("write", self.updateDriftCorrectionSettings)
        self.view.automatedMicroscopySettingstab.driftcorrection_552.trace_add("write", self.updateDriftCorrectionSettings)
        self.view.automatedMicroscopySettingstab.driftcorrection_594.trace_add("write", self.updateDriftCorrectionSettings)
        self.view.automatedMicroscopySettingstab.driftcorrection_640.trace_add("write", self.updateDriftCorrectionSettings)
        self.view.automatedMicroscopySettingstab.driftcorrection_LED.trace_add("write", self.updateDriftCorrectionSettings)

        #define some parameters
        self.current_laser = "488"

    def run(self):
        """
        Run the Tkinter Gui in the main loop
        :return:
        """
        self.root.title("Multi-scale microscope V1")
        self.root.geometry("800x600")
        self.root.resizable(width=False, height=False)
        #self.automatically_update_stackbuffer()
        self.root.mainloop()

    def close(self):
        self.model.LED_voltage.setconstantvoltage(0)
        self.model.close()

    def updateLowResLaserParameters(self, var, indx, mode):
        """
        update the laser power
        """
        # get laser power from GUI and construct laser power setting array
        # multiply with 5 here as the laser is modulated within 0 to 5 V
        voltage488_LR = self.view.runtab.laser488_percentage_LR.get() * 5 / 100.
        voltage552_LR = self.view.runtab.laser552_percentage_LR.get() * 5 / 100.
        voltage594_LR = self.view.runtab.laser594_percentage_LR.get() * 5 / 100.
        voltage640_LR = self.view.runtab.laser640_percentage_LR.get() * 5 / 100.
        power_settings_LR = [voltage488_LR, voltage552_LR, voltage594_LR, voltage640_LR]

        # change laser power
        self.model.lowres_laserpower=power_settings_LR

    def updateHighResLaserParameters(self, var, indx, mode):
        """
        update the laser power
        """
        voltage488_HR = self.view.runtab.laser488_percentage_HR.get() * 5 / 100.
        voltage552_HR = self.view.runtab.laser552_percentage_HR.get() * 5 / 100.
        voltage594_HR = self.view.runtab.laser594_percentage_HR.get() * 5 / 100.
        voltage640_HR = self.view.runtab.laser640_percentage_HR.get() * 5 / 100.
        power_settings_HR = [voltage488_HR, voltage552_HR, voltage594_HR, voltage640_HR]
        self.model.highres_laserpower=power_settings_HR

    def updateSlitParameters(self, var, indx, mode):
        # set the low resolution and high-resolution slit openings
        self.model.slitopening_lowres = self.view.advancedSettingstab.slit_lowres.get()
        self.model.slitopening_highres = self.view.advancedSettingstab.slit_highres.get()

    def update_mSPIMvoltage(self, var, indx, mode):
        # set the voltage for the mSPIM mirror
        voltage = self.view.advancedSettingstab.adv_settings_mSPIMvoltage.get()
        print(voltage)
        if voltage > 0 and voltage < NI_board_parameters.max_mSPIM_constant:
            self.model.mSPIMmirror_voltage.setconstantvoltage(voltage)

    def updateExposureParameters(self, var, indx, mode):
        # exposure time

        if self.view.runtab.cam_lowresExposure.get()>5:
            self.model.exposure_time_LR = self.view.runtab.cam_lowresExposure.get()
        if self.view.runtab.cam_highresExposure.get()>5:
            self.model.exposure_time_HR = self.view.runtab.cam_highresExposure.get()  # set exposure time
        print("updated exposure time")

    def update_stack_aq_parameters(self, var, indx, mode):
        #advanced stack acquisition parameters from advanced settings tab
        self.model.delay_cameratrigger = self.view.advancedSettingstab.stack_aq_camera_delay.get()/1000 #divide by 1000 - from ms to seconds
        self.model.highres_planespacing = int(self.view.runtab.stack_aq_plane_spacing_highres.get() * 1000000)
        self.model.lowres_planespacing = int(self.view.runtab.stack_aq_plane_spacing_lowres.get() * 1000000)
        self.model.stack_nbplanes_highres = int(self.view.runtab.stack_aq_numberOfPlanes_highres.get())
        self.model.stack_nbplanes_lowres = int(self.view.runtab.stack_aq_numberOfPlanes_lowres.get())
        self.model.slow_velocity = self.view.advancedSettingstab.stack_aq_stage_velocity.get()
        self.model.slow_acceleration = self.view.advancedSettingstab.stack_aq_stage_acceleration.get()

        print("stack acquisition settings updated")

    def update_ASLMParameters(self, var, indx, mode):
        # get remote mirror voltage from GUI and update model parameter, also check for boundaries
        minVol = ASLM_parameters.remote_mirror_minVol
        maxVol = ASLM_parameters.remote_mirror_maxVol
        #
        try:
            interval = self.view.advancedSettingstab.ASLM_volt_interval.get() / 1000
            middle_range488 = self.view.advancedSettingstab.ASLM_volt_middle488.get() / 1000
            middle_range552 = self.view.advancedSettingstab.ASLM_volt_middle552.get() / 1000
            middle_range594 = self.view.advancedSettingstab.ASLM_volt_middle594.get() / 1000
            middle_range640 = self.view.advancedSettingstab.ASLM_volt_middle640.get() / 1000
            middle_range = [middle_range488, middle_range552, middle_range594, middle_range640]
            print(interval)
        except:
            interval = 0
            middle_range = 10
        #

        setvoltage_first = 0
        setvoltage_second = 0
        if self.view.advancedSettingstab.ASLM_voltageDirection.get() == 'highTolow':
            setvoltage_first = [x + (interval / 2) for x in middle_range]
            setvoltage_second = [x - (interval / 2) for x in middle_range]
        else:
            setvoltage_first = [x - (interval / 2) for x in middle_range]
            setvoltage_second = [x + (interval / 2) for x in middle_range]

        #check boundaries
        setvoltage_first = np.minimum(maxVol, np.maximum(setvoltage_first, minVol))
        setvoltage_second = np.minimum(maxVol, np.maximum(setvoltage_second, minVol))

        self.model.ASLM_from_Volt = setvoltage_first
        self.model.ASLM_to_Volt = setvoltage_second
        # display calculated voltages
        #self.view.advancedSettingstab.voltage_minIndicator.config(text=str(round(self.model.ASLM_from_Volt,5)))
        #self.view.advancedSettingstab.voltage_maxIndicator.config(text=str(round(self.model.ASLM_to_Volt, 5)))
        print(self.model.ASLM_from_Volt)
        print(self.model.ASLM_to_Volt)
        self.model.ASLM_currentVolt = min(maxVol, max(minVol, self.view.advancedSettingstab.ASLM_volt_current.get()/1000))

        # update the ASLM alignment settings
        self.model.ASLM_alignmentOn = self.view.advancedSettingstab.ASLM_alignmentmodeOn.get()
        print(self.model.ASLM_alignmentOn)

        #update scanwidth
        self.model.ASLM_scanWidth = self.view.advancedSettingstab.ASLM_scanWidth.get()

    def updateDriftCorrectionSettings(self, var, indx, mode):
        '''
        update settings of the drift correction
        :return: settings updated in model
        '''
        #determine whether drift correction is active; and on which channel
        print("drift correction settings updated")
        self.model.drift_correctionOnHighRes = self.view.automatedMicroscopySettingstab.drift_correction_highres.get()  # parameter whether high res drift correction is enabled
        self.model.drift_correctionOnLowRes = self.view.automatedMicroscopySettingstab.drift_correction_lowres.get()  # parameter whether low res drift correction is enabled
        self.model.drift_which_channels = [self.view.automatedMicroscopySettingstab.driftcorrection_488.get(),
                                           self.view.automatedMicroscopySettingstab.driftcorrection_552.get(),
                                           self.view.automatedMicroscopySettingstab.driftcorrection_594.get(),
                                           self.view.automatedMicroscopySettingstab.driftcorrection_640.get(),
                                           self.view.automatedMicroscopySettingstab.driftcorrection_LED.get()]

        if self.view.automatedMicroscopySettingstab.driftcorrection_LED.get()==1:
            self.model.drift_transmission = 1
        else:
            self.model.drift_transmission = 0


    def updateGUItext(self):
        '''
        update text labes in GUI here
        :return:
        '''
        self.model.currentFPS #todo

    def changeROI(self,event):
        '''
        change the ROI - options ('Full Chip', '1024x1024', '512x512', '256x256', 'Custom')

        :return:
        '''
        #which ROI selected
        lowresstartx = 0
        lowresstarty = 0
        if self.view.runtab.roi_whichresolution.get()=='on': #low-resolution
            if self.model.continue_preview_lowres == False:
                if self.view.runtab.roi_ac_settings_type.get() == 'Full Chip':
                    self.model.current_lowresROI_height = Camera_parameters.LR_height_pixel
                    self.model.current_lowresROI_width = Camera_parameters.LR_width_pixel
                    lowresstartx = 0
                    lowresstarty = 0
                    self.model.lowres_camera.set_imageroi(lowresstartx,lowresstarty, Camera_parameters.LR_width_pixel, Camera_parameters.LR_height_pixel)
                if self.view.runtab.roi_ac_settings_type.get() == '1024x1024':
                    self.model.current_lowresROI_height = 1024
                    self.model.current_lowresROI_width = 1024
                    lowresstartx = int(Camera_parameters.LR_width_pixel/2)-512
                    lowresstarty = int(Camera_parameters.LR_height_pixel/2)-512
                    self.model.lowres_camera.set_imageroi(lowresstartx,lowresstarty, 1024, 1024)
                if self.view.runtab.roi_ac_settings_type.get() == '512x512':
                    self.model.current_lowresROI_height = 512
                    self.model.current_lowresROI_width = 512
                    lowresstartx = int(Camera_parameters.LR_width_pixel/2)-256
                    lowresstarty = int(Camera_parameters.LR_height_pixel/2)-256
                    self.model.lowres_camera.set_imageroi(lowresstartx,lowresstarty, 512, 512)
                if self.view.runtab.roi_ac_settings_type.get() == 'Usual':
                    self.model.current_lowresROI_height = 2960
                    self.model.current_lowresROI_width = 3816
                    lowresstartx = 620
                    lowresstarty = 0
                    self.model.lowres_camera.set_imageroi(620,0, 3816, 2960)
                if self.view.runtab.roi_ac_settings_type.get() == '256x256':
                    self.model.current_lowresROI_height = 256
                    self.model.current_lowresROI_width = 256
                    lowresstartx = int(Camera_parameters.LR_width_pixel / 2) - 128
                    lowresstarty = int(Camera_parameters.LR_height_pixel / 2) - 128
                    self.model.lowres_camera.set_imageroi(lowresstartx,lowresstarty, 256, 256)
                if self.view.runtab.roi_ac_settings_type.get() == 'Custom':
                    self.model.current_lowresROI_height = self.view.runtab.roi_height.get()
                    self.model.current_lowresROI_width = self.view.runtab.roi_width.get()
                    lowresstartx = self.view.runtab.roi_startX.get()
                    lowresstarty = self.view.runtab.roi_startY.get()
                    self.model.lowres_camera.set_imageroi(lowresstartx,lowresstarty, self.model.current_lowresROI_width, self.model.current_lowresROI_height)

            #update calibration of drift correction module
            x1_width = lowresstartx
            x2_width = Camera_parameters.LR_width_pixel - self.model.current_lowresROI_width - lowresstartx
            shiftvector_width = round((x2_width - x1_width)/2)
            self.drift_correctionmodule.calibration_width = Camera_parameters.low_to_highres_calibration_width + shiftvector_width
            print("drift correction calibration width " + str(self.drift_correctionmodule.calibration_width))
            x1_height = lowresstarty
            x2_height = Camera_parameters.LR_height_pixel - self.model.current_lowresROI_height - lowresstarty
            shiftvector_height = round((x2_height - x1_height)/2)
            self.drift_correctionmodule.calibration_height = Camera_parameters.low_to_highres_calibration_height + shiftvector_height
            print("drift correction calibration height " + str(self.drift_correctionmodule.calibration_height))

        else: #change high-res ROI
            if self.model.continue_preview_highres == False:
                if self.view.runtab.roi_ac_settings_type.get() == 'Full Chip':
                    self.model.current_highresROI_height = Camera_parameters.HR_height_pixel
                    self.model.current_highresROI_width = Camera_parameters.HR_width_pixel
                    self.model.highres_camera.set_imageroi(0, 0, Camera_parameters.HR_width_pixel,
                                                          Camera_parameters.HR_height_pixel)
                if self.view.runtab.roi_ac_settings_type.get() == '1024x1024':
                    self.model.current_highresROI_height = 1024
                    self.model.current_highresROI_width = 1024
                    startx = int(Camera_parameters.HR_width_pixel / 2) - 512
                    starty = int(Camera_parameters.HR_height_pixel / 2) - 512
                    self.model.highres_camera.set_imageroi(startx, starty, 1024, 1024)
                if self.view.runtab.roi_ac_settings_type.get() == '512x512':
                    self.model.current_highresROI_height = 512
                    self.model.current_highresROI_width = 512
                    startx = int(Camera_parameters.HR_width_pixel/2)-256
                    starty = int(Camera_parameters.HR_height_pixel/2)-256
                    self.model.highres_camera.set_imageroi(startx, starty, 512, 512)
                if self.view.runtab.roi_ac_settings_type.get() == '256x256':
                    self.model.current_highresROI_height = 256
                    self.model.current_highresROI_width = 256
                    startx = int(Camera_parameters.HR_width_pixel / 2) - 128
                    starty = int(Camera_parameters.HR_height_pixel / 2) - 128
                    self.model.highres_camera.set_imageroi(startx, starty, 256, 256)
                if self.view.runtab.roi_ac_settings_type.get() == 'Usual':
                    self.model.current_highresROI_height = 1024
                    self.model.current_highresROI_width = 2048
                    self.model.highres_camera.set_imageroi(0, 512, 2048, 1024)
                if self.view.runtab.roi_ac_settings_type.get() == 'Custom':
                    self.model.current_highresROI_height = self.view.runtab.roi_height.get()
                    self.model.current_highresROI_width = self.view.runtab.roi_width.get()
                    startx = self.view.runtab.roi_startX.get()
                    starty = self.view.runtab.roi_startY.get()
                    self.model.highres_camera.set_imageroi(startx, starty, self.model.current_highresROI_width, self.model.current_highresROI_height)

    def run_lowrespreview(self, event):
        '''
        Runs the execution of a low resolution preview.
        Required:
        change mirror, start preview, set continue_preview_highres to True.
        '''

        #end highres preview
        self.model.continue_preview_highres = False
        self.view.runtab.bt_preview_highres.config(relief="raised")

        if self.model.continue_preview_lowres == False:

            #change mirror position/slit position
            self.model.changeHRtoLR()

            # set parameter that you run a preview
            self.model.continue_preview_lowres = True
            #self.model.laserOn = self.current_laser

            #set button layout - sunken relief
            def set_button():
                time.sleep(0.002)
                self.view.runtab.bt_preview_lowres.config(relief="sunken")
            ct.ResultThread(target=set_button).start()

            #run preview with given parameters
            self.model.preview_lowres()
            print("running lowres preview")

    def run_highrespreview(self, event):
        '''
        Runs the execution of a high resolution preview.
        Required:
        change mirror, set exposure time, start preview, set continue_preview_highres to True.
        '''

        #end highres preview
        self.model.continue_preview_lowres = False
        self.view.runtab.bt_preview_lowres.config(relief="raised")

        if self.model.continue_preview_highres == False:

            # change mirror position/slit position
            self.model.changeLRtoHR()

            # set parameter that you run a preview
            self.model.continue_preview_highres = True
            #self.model.laserOn = self.current_laser

            #set button layout - sunken relief
            def set_buttonHR():
                time.sleep(0.002)
                self.view.runtab.bt_preview_highres.config(relief="sunken")
            ct.ResultThread(target=set_buttonHR).start()

            #run preview with given parameters

            #ASLM or static light-sheet mode
            if self.view.runtab.cam_highresMode.get()=='SPIM Mode':
                self.model.preview_highres_static()
                print("running high res static preview")
            else:
                self.model.preview_highres_ASLM()
                print("running high res ASLM preview")

    def updatepreview(self, var, indx, mode):
        '''
        Updates preview functionalities: auto-scaling of intensity values
        '''
        if self.view.runtab.preview_autoIntensity.get() == 1:
            self.model.autoscale_preview = 1
            print("updated ---------------------------------------1")
        else:
            self.model.autoscale_preview =0

    def run_stop_preview(self, event):
        '''
        Stops an executing preview and resets the profile of the preview buttons that were sunken after starting a preview
        '''
        if self.model.continue_preview_lowres == True:
            self.model.continue_preview_lowres =False
            self.view.runtab.preview_change(self.view.runtab.bt_preview_lowres)

        if self.model.continue_preview_highres == True:
            self.model.continue_preview_highres = False
            self.view.runtab.preview_change(self.view.runtab.bt_preview_highres)

    def movestage(self, var,indx, mode):
        """
        moves the stage to a certain position
        """
        #get positions from GUI and constract position array "moveToPosition"
        lateralPosition = self.view.stagessettingstab.stage_moveto_lateral.get() * 1000000000
        updownPosition =self.view.stagessettingstab.stage_moveto_updown.get() * 1000000000
        axialPosition =self.view.stagessettingstab.stage_moveto_axial.get() * 1000000000
        anglePosition =self.view.stagessettingstab.stage_moveto_angle.get() * 1000000
        moveToPosition = [axialPosition, lateralPosition, updownPosition, anglePosition]

        #check not to exceed limits
        moveToPosition = self.model.check_movementboundaries(moveToPosition)

        #move
        self.model.move_to_position(moveToPosition)

    def movestageToPosition(self, event):
        """
        moves the stage to a saved position, indicated by a field in the GUI
        """
        position = self.view.stagessettingstab.stage_move_to_specificposition.get()
        print(position)

        if self.view.stagessettingstab.move_to_specific_pos_resolution.get() == "on":
            print("move:")
            for line in range(len(self.view.stagessettingstab.stage_PositionList)):
                savedpos = int(self.view.stagessettingstab.stage_PositionList[line][0])
                if savedpos == position:
                    #set positions in the moving panel:
                    self.view.stagessettingstab.stage_moveto_lateral.set(self.view.stagessettingstab.stage_PositionList[line][1])
                    self.view.stagessettingstab.stage_moveto_updown.set(self.view.stagessettingstab.stage_PositionList[line][2])
                    self.view.stagessettingstab.stage_moveto_axial.set(self.view.stagessettingstab.stage_PositionList[line][3])
                    self.view.stagessettingstab.stage_moveto_angle.set(self.view.stagessettingstab.stage_PositionList[line][4])
                    #move to these positions:
                    xpos = int(float(self.view.stagessettingstab.stage_PositionList[line][1]) * 1000000000)
                    ypos = int(float(self.view.stagessettingstab.stage_PositionList[line][2]) * 1000000000)
                    zpos = int(float(self.view.stagessettingstab.stage_PositionList[line][3]) * 1000000000)
                    angle = int(float(self.view.stagessettingstab.stage_PositionList[line][4]) * 1000000)
                    current_position = [zpos, xpos, ypos, angle]
                    self.model.move_to_position(current_position)
        else:
            for line in range(len(self.view.stagessettingstab.stage_highres_PositionList)):
                savedpos = int(self.view.stagessettingstab.stage_highres_PositionList[line][0])
                if savedpos == position:
                    self.view.stagessettingstab.stage_moveto_lateral.set(self.view.stagessettingstab.stage_highres_PositionList[line][1])
                    self.view.stagessettingstab.stage_moveto_updown.set(self.view.stagessettingstab.stage_highres_PositionList[line][2])
                    self.view.stagessettingstab.stage_moveto_axial.set(self.view.stagessettingstab.stage_highres_PositionList[line][3])
                    self.view.stagessettingstab.stage_moveto_angle.set(self.view.stagessettingstab.stage_highres_PositionList[line][4])

                    xpos = int(float(self.view.stagessettingstab.stage_highres_PositionList[line][1]) * 1000000000)
                    ypos = int(float(self.view.stagessettingstab.stage_highres_PositionList[line][2]) * 1000000000)
                    zpos = int(float(self.view.stagessettingstab.stage_highres_PositionList[line][3]) * 1000000000)
                    angle = int(float(self.view.stagessettingstab.stage_highres_PositionList[line][4]) * 1000000)
                    current_position = [zpos, xpos, ypos, angle]
                    self.model.move_to_position(current_position)

    def slit_opening_move(self, var,indx, mode):
        """
        changes the slit opening
        """
        currentslitopening = self.view.advancedSettingstab.slit_currentsetting.get()
        self.model.move_adjustableslit(currentslitopening)
        self.view.advancedSettingstab.slit_currentsetting.set(currentslitopening)


    def changefilter(self, event, laser):
        """
        changes the filter to the specified one by the laser active
        """
        print("change filter to laser: " + laser)
        if laser == '488':
            self.model.LED_voltage.setconstantvoltage(0)
            self.model.filterwheel.set_filter('515-30-25', wait_until_done=False)
            self.model.current_laser = NI_board_parameters.laser488
        if laser == '552':
            self.model.LED_voltage.setconstantvoltage(0)
            self.model.filterwheel.set_filter('572/20-25', wait_until_done=False)
            self.model.current_laser = NI_board_parameters.laser552
        if laser == '594':
            self.model.LED_voltage.setconstantvoltage(0)
            self.model.filterwheel.set_filter('615/20-25', wait_until_done=False)
            self.model.current_laser = NI_board_parameters.laser594
        if laser == '640':
            self.model.LED_voltage.setconstantvoltage(0)
            self.model.filterwheel.set_filter('676/37-25', wait_until_done=False)
            self.model.current_laser = NI_board_parameters.laser640
        if laser =='LED':
            self.model.filterwheel.set_filter('515-30-25', wait_until_done=False)
            self.model.LED_voltage.setconstantvoltage(4)
            self.model.current_laser = NI_board_parameters.led


    def updatefilename(self):
        """
        construct the filename used to save data, based on the information from the GUI
        """
        parentdir = FileSave_parameters.parentdir

        modelorganism = self.view.welcometab.welcome_modelorganism.get()
        date = dt.datetime.now().strftime("%Y%m%d")
        username = self.view.welcometab.welcome_username.get()

        foldername = date + "_" + modelorganism
        if username == "Stephan Daetwyler":
            foldername = date + "_Daetwyler_" + modelorganism
        if username == "Reto Fiolka":
            foldername = date + "_Fiolka_" + modelorganism
        if username == "Bo-Jui Chang":
            foldername = date + "_Chang_" + modelorganism
        if username == "Dagan Segal":
            foldername = date + "_Segal_" + modelorganism

        self.parentfolder = os.path.join(parentdir, foldername)

    def acquire_stack(self, event):
        """
        start a stack acquisition thread
        """
        self.model.abortStackFlag = 0
        ct.ResultThread(target=self.acquire_stack_task).start()

    def abort_stack(self, event):
        """
        set flag to abort stack acquisition
        """
        self.model.abortStackFlag = 1


    def acquire_stack_task(self):
        """
        acquire a stack acquisition - processes in thread (to not stop GUI from working)
        """
        self.view.runtab.stack_aq_bt_run_stack.config(relief="sunken")
        self.view.update()
        self.updatefilename()

        #stop all potential previews
        self.model.continue_preview_lowres = False
        self.model.continue_preview_highres = False

        #overwrite abortStackFlag if pressed before
        self.model.abortStackFlag = 0

        #some parameters.
        self.model.displayImStack = self.view.runtab.stack_aq_displayON.get() #whether to display images during stack acq or not

        #set driftcorrection settings - set completed array back to zero and update position lists in module
        self.model.driftcorrectionmodule.completed=np.zeros(len(self.view.stagessettingstab.stage_highres_PositionList))
        self.model.driftcorrectionmodule.highres_positionList = copy.deepcopy(self.view.stagessettingstab.stage_highres_PositionList)
        self.model.driftcorrectionmodule.lowres_positionList = copy.deepcopy(self.view.stagessettingstab.stage_PositionList)

        #save acquistition parameters and construct file name to save (only if not time-lapse)
        stackfilepath = self.parentfolder
        if self.continuetimelapse != 0:

            # generate file path
            nbfiles_folder = len(glob.glob(os.path.join(self.parentfolder, 'Experiment*')))
            newfolderind = nbfiles_folder + 1

            # catch errors when deleting Experiment names before
            def set_experiment_name(parentfolder, experimentnumber):
                experiment_name = "Experiment" + f'{experimentnumber:04}'
                filepath = os.path.join(parentfolder, experiment_name)
                isExist = os.path.exists(filepath)

                if isExist:
                    experiment_name = set_experiment_name(parentfolder, experimentnumber + 1)

                return experiment_name

            experiment_name = set_experiment_name(self.parentfolder, newfolderind)

            #write acquisition parameters
            filepath_write_acquisitionParameters = os.path.join(self.parentfolder, experiment_name)
            try:
                print("filepath : " + filepath_write_acquisitionParameters)
                os.makedirs(filepath_write_acquisitionParameters)
            except OSError as error:
                print("File writing error")

            #write parameters in a thread
            def write_paramconfig():
                lowresroi = [self.model.current_lowresROI_width, self.model.current_lowresROI_height]
                highresroi = [self.model.current_highresROI_width, self.model.current_highresROI_height]
                self.paramwriter.write_to_textfile(
                    os.path.join(filepath_write_acquisitionParameters, 'Experiment_settings.txt'), lowresroi,
                    highresroi)
                print("parameters saved")
            ct.ResultThread(target=write_paramconfig).start()

            #set timepoint = 0 to be consistent with time-lapse acquisitions
            self.model.current_timepointstring = "t00000"
            stackfilepath = os.path.join(self.parentfolder, experiment_name, "t00000")
            self.model.experimentfilepath = os.path.join(self.parentfolder, experiment_name)
            print(stackfilepath)

            #no need for drift correction if single stack is acquired
            self.model.drift_correctionOnHighRes = 0  # parameter whether high res drift correction is enabled
            self.model.drift_correctionOnLowRes = 0  # parameter whether low res drift correction is enabled
        else:
            stackfilepath = self.current_timelapse_filepath

        ########-------------------------------------------------------------------------------------------------------
        #start low resolution stack acquisition
        if self.view.runtab.stack_aq_lowResCameraOn.get():
            print("acquiring low res stack")

            # change mirror position/slit position
            self.model.changeHRtoLR()

            #iterate over all positions in the low res Position list
            positioniter = -1
            for line in range(len(self.view.stagessettingstab.stage_PositionList)):
                #get current position from list
                current_startposition = self.view.stagessettingstab.stage_PositionList[line]
                positioniter = positioniter + 1

                # filepath
                current_folder = os.path.join(stackfilepath, "low_stack" + f'{positioniter:03}')

                #update info for filepath for projections and drift correction
                self.model.current_region =  "low_stack" + f'{positioniter:03}'
                self.model.current_PosNumber = positioniter

                try:
                    print("filepath : " + current_folder)
                    os.makedirs(current_folder)
                except OSError as error:
                    print("File writing error")

                #start stackstreaming
                which_channels = [self.view.runtab.stack_aq_488onLowRes.get(), self.view.runtab.stack_aq_552onLowRes.get(),
                                  self.view.runtab.stack_aq_594onLowRes.get(), self.view.runtab.stack_aq_640onLowRes.get(),
                                  self.view.runtab.stack_aq_LEDonLowRes.get()]
                self.model.stack_acquisition_master(current_folder, current_startposition, which_channels, "low")


        #################-----------------------------------------------------------------------------------------------
        #wait for all drift correction positions to be calculated before proceeding to high resolution imaging
        if self.continuetimelapse == 0:
            #call here drift correction if based on low resolution imaging
            if self.view.automatedMicroscopySettingstab.drift_correction_lowres.get()==1:

                print("Wait until drift correction is calculated")
                while np.sum(self.model.driftcorrectionmodule.completed) != len(self.model.driftcorrectionmodule.completed):
                    time.sleep(0.05)
                    if self.continuetimelapse == 1:
                        break

                #update stage position list by taking latest list from driftcorrectionmodule
                self.view.stagessettingstab.stage_highres_PositionList = self.model.driftcorrectionmodule.highres_positionList


        ########-------------------------------------------------------------------------------------------------------
        # start high resolution stack acquisition
        #high resolution list
        if self.view.runtab.stack_aq_highResCameraOn.get():
            # change mirror position/slit position
            self.model.changeLRtoHR()

            print("acquiring high res stack")
            for line in range(len(self.view.stagessettingstab.stage_highres_PositionList)):
                #get position
                currentposition = self.view.stagessettingstab.stage_highres_PositionList[line]

                #define highresolution stack file path label by label position in file position tree
                #(can be updated e.g. if you have automatic detection during timelapse)
                pos_label = int(self.view.stagessettingstab.stage_highres_PositionList[line][5])

                # filepath
                current_folder = os.path.join(stackfilepath, "high_stack_" + f'{pos_label:03}')
                try:
                    print("filepath : " + current_folder)
                    os.makedirs(current_folder)
                except OSError as error:
                    print("File writing error")

                # update info for filepath for projections and drift correction
                self.model.current_region = "high_stack_" + f'{pos_label:03}'
                self.model.current_PosNumber = pos_label

                # start stackstreaming
                which_channels = [self.view.runtab.stack_aq_488onHighRes.get(), self.view.runtab.stack_aq_552onHighRes.get(),
                                  self.view.runtab.stack_aq_594onHighRes.get(), self.view.runtab.stack_aq_640onHighRes.get(),
                                  self.view.runtab.stack_aq_LEDonHighRes.get()]

                if self.view.runtab.cam_highresMode.get()=="SPIM Mode":
                    self.model.stack_acquisition_master(current_folder, currentposition, which_channels, "highSPIM")
                else:
                    self.model.stack_acquisition_master(current_folder, currentposition, which_channels, "highASLM")

        self.view.runtab.stack_aq_bt_run_stack.config(relief="raised")
        self.model.LED_voltage.setconstantvoltage(0)


    def acquire_timelapse(self, event):
        """
        start a time-lapse acquisition thread, called from GUI (otherwise it freezes)
        """

        #update GUI
        self.view.runtab.updateTimesTimelapse()
        self.view.update_idletasks()
        self.view.update()
        self.view.runtab.timelapse_aq_progressbar.config(maximum=self.view.runtab.timelapse_aq_nbTimepoints-1)
        self.updatefilename()

        # set button layout - sunken relief
        def set_buttonTL():
            time.sleep(0.002)
            self.view.runtab.timelapse_aq_bt_run_timelapse.config(relief="sunken")
        ct.ResultThread(target=set_buttonTL).start()

        # stop all potential previews
        self.model.continue_preview_lowres = False
        self.model.continue_preview_highres = False

        self.continuetimelapse = 0
        print("acquiring timelapse")

        #(1) NOTE: You cannot use a While loop here as it makes the Tkinter mainloop freeze - put the time-lapse instead into a thread
        #(2) where you can run while loops etc.
        self.timelapse_thread = Thread(target=self.run_timelapse)
        self.timelapse_thread.start()
        #after that main loop continues


    def run_timelapse(self):
        """
        thread that controls time-lapse, started from function acquire_timelapse, which is called from GUI(self, event):
        """

        # generate file path
        nbfiles_folder = len(glob.glob(os.path.join(self.parentfolder, 'Experiment*')))
        newfolderind = nbfiles_folder + 1

        # generate experiment name - the recursive loop is required to catch errors when deleting Experiment names before
        def set_experiment_name(parentfolder, experimentnumber):
            experiment_name = "Experiment" + f'{experimentnumber:04}'
            filepath = os.path.join(parentfolder, experiment_name)
            isExist = os.path.exists(filepath)

            if isExist:
                experiment_name = set_experiment_name(parentfolder, experimentnumber+1)

            return experiment_name

        experiment_name = set_experiment_name(self.parentfolder, newfolderind)
        experimentpath = os.path.join(self.parentfolder, experiment_name)

        # generate filepath and folder for writing acquisition parameters
        filepath_write_acquisitionParameters = os.path.join(self.parentfolder, experiment_name)
        try:
            print("filepath : " + filepath_write_acquisitionParameters)
            os.makedirs(filepath_write_acquisitionParameters)
        except OSError as error:
            print("File writing error")

        # write acquisition parameters in a thread
        def write_paramconfigtimelapse():
            lowresroi = [self.model.current_lowresROI_width, self.model.current_lowresROI_height]
            highresroi = [self.model.current_highresROI_width, self.model.current_highresROI_height]
            self.paramwriter.write_to_textfile(os.path.join(filepath_write_acquisitionParameters, 'Experiment_settings.txt'),lowresroi, highresroi)
            print("parameters saved")
        ct.ResultThread(target=write_paramconfigtimelapse).start()

        #driftcorrection - reinitalize image repository so that not old images are used for comparison and set logfolder path
        self.model.driftcorrectionmodule.ImageRepo.reset()
        try:
            logfolderpath = os.path.join(experimentpath, 'driftcorrection_log')
            print("drift correction logfolder : " + logfolderpath)
            self.model.driftcorrectionmodule.logfolder = logfolderpath
            os.makedirs(logfolderpath)
        except OSError as error:
            print("Error generating logfolder")

        ###run timelapse, starting at timepoint 0
        for timeiter in range(0, self.view.runtab.timelapse_aq_nbTimepoints):
            t0 = time.perf_counter()

            #generate timepoint strings
            numStr = str(timeiter).zfill(5)
            pastStr = str(timeiter-1).zfill(5)
            self.model.current_timepointstring = "t" + numStr
            self.model.past_timepointstring = "t" + pastStr
            self.current_timelapse_filepath = os.path.join(experimentpath, self.model.current_timepointstring)
            self.model.experimentfilepath = experimentpath #for projection

            #update GUI about timelapse progress
            self.view.runtab.timelapse_aq_progress.set(timeiter)
            self.view.runtab.timelapse_aq_progressindicator.config(text=str(timeiter+1) +" of " + str(self.view.runtab.timelapse_aq_nbTimepoints))

            timeinterval = self.view.runtab.timelapse_aq_timeinterval_min.get()*60 + self.view.runtab.timelapse_aq_timeinterval_seconds.get()
            print("time interval:"  + str(timeinterval))

            ## stop time-lapse acquisition if you stop it
            if self.continuetimelapse == 1:
                break  # Break while loop when stop = 1

            #start stack acquisition and wait for it to finish before continuing with next stack acquisition
            stackacquisitionthread = ct.ResultThread(target=self.acquire_stack_task).start()
            stackacquisitionthread.get_result()

            #calculate the time until next stack acquisition starts and wait for the time counter to progress
            t1 = time.perf_counter() - t0
            totaltime = self.view.runtab.timelapse_aq_timeinterval_min.get() * 60 + self.view.runtab.timelapse_aq_timeinterval_seconds.get()

            remaining_waittime = 1
            while (remaining_waittime>0) and (self.continuetimelapse == 0):
                t1 = time.perf_counter() - t0
                remaining_waittime = totaltime - t1

        self.continuetimelapse = 1
        self.view.runtab.timelapse_aq_bt_run_timelapse.config(relief="raised")
        self.view.update()


    def abort_timelapse(self,event):
        self.continuetimelapse = 1


#enable keyboard movements ---------------------------------------------------------------------------------------------
    def enable_keyboard_movement(self, event):
        self.root.bind("<Key>", self.key_pressed)
        self.root.update()

    def disable_keyboard_movement(self, event):
        self.root.unbind("<Key>")
        self.root.update()

    def key_pressed(self, event):
        print(event.keysym)
        if event.char == "w" or event.keysym =="Up":
            self.view.stagessettingstab.change_currentposition(self.view.stagessettingstab.stage_moveto_updown, 1)
            self.view.stagessettingstab.stage_last_key.set("w")

        if event.char == "s" or event.keysym =="Down":
            self.view.stagessettingstab.change_currentposition(self.view.stagessettingstab.stage_moveto_updown, -1)
            self.view.stagessettingstab.stage_last_key.set("s")

        if event.char =="a" or event.keysym =="Left":
            self.view.stagessettingstab.change_currentposition(self.view.stagessettingstab.stage_moveto_lateral, -1)
            self.view.stagessettingstab.stage_last_key.set("a")

        if event.char == "d" or event.keysym =="Right":
            self.view.stagessettingstab.change_currentposition(self.view.stagessettingstab.stage_moveto_lateral, 1)
            self.view.stagessettingstab.stage_last_key.set("d")

        if event.char == "q":
            self.view.stagessettingstab.change_currentposition(self.view.stagessettingstab.stage_moveto_axial, 1)
            self.view.stagessettingstab.stage_last_key.set("q")

        if event.char == "e":
            self.view.stagessettingstab.change_currentposition(self.view.stagessettingstab.stage_moveto_axial, -1)
            self.view.stagessettingstab.stage_last_key.set("e")


if __name__ == '__main__':
    c = MultiScale_Microscope_Controller()
    c.run()
    c.close()





