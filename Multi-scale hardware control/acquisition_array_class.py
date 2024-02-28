import numpy as np
from auxiliary_code.constants import NI_board_parameters
from auxiliary_code.constants import Camera_parameters

class acquisition_arrays:
    """
    generate the voltage curves needed for the acquisitions on the NI board
    returns the arrays.
    """
    def __init__(self, model):
        self.model = model

    def get_lowres_preview_array(self):
        # define array for laser
        basic_unit = np.zeros((self.model.ao.s2p(0.3), NI_board_parameters.ao_nchannels), np.dtype(np.float64))

        if self.model.current_laser>0: #no-laser for LED
            basic_unit[:, self.model.current_laser] = 4.  # set TTL signal of the right laser to 4
        basic_unit[:, NI_board_parameters.voicecoil] = self.model.ASLM_staticLowResVolt  # set voltage of remote mirror

        if self.model.ASLM_alignmentOn == 1:
            basic_unit[:, NI_board_parameters.voicecoil] = self.model.ASLM_currentVolt  # set voltage of remote mirror

        return basic_unit

    def get_highres_preview_array(self):
        # define array for laser
        min_time = max(self.model.exposure_time_HR / 1000, 3)
        basic_unit = np.zeros((self.model.ao.s2p(min_time), NI_board_parameters.ao_nchannels),
                              np.dtype(np.float64))

        if self.model.current_laser > 0:#no-laser for LED
            basic_unit[:, self.model.current_laser] = 4.  # set TTL signal of the right laser to 4
        basic_unit[:, NI_board_parameters.voicecoil] = self.model.ASLM_staticHighResVolt  # set voltage of remote mirror

        if self.model.ASLM_alignmentOn == 1:
            basic_unit[:, NI_board_parameters.voicecoil] = self.model.ASLM_currentVolt  # set voltage of remote mirror

        return basic_unit

    def get_highresASLM_preview_array(self):
        # define array size
        totallength = self.model.ao.s2p(0.001 + self.model.ASLM_acquisition_time / 1000 + 0.02)
        basic_unit = np.zeros((totallength, NI_board_parameters.ao_nchannels), np.dtype(np.float64))

        if self.model.current_laser > 0:  # no-laser for LED
            basic_unit[self.model.ao.s2p(0.001):totallength, self.model.current_laser] = 4

        basic_unit[self.model.ao.s2p(0.001): self.model.ao.s2p(0.002), NI_board_parameters.highres_camera] = 4.  # high-res camera

        ASLM_from = self.model.ASLM_from_Volt[self.model.current_laser - NI_board_parameters.adjustmentfactor]
        ASLM_to = self.model.ASLM_to_Volt[self.model.current_laser - NI_board_parameters.adjustmentfactor]

        sawtooth_array = np.zeros(totallength, np.dtype(np.float64))
        sawtooth_array[:] = ASLM_from
        goinguppoints = self.model.ao.s2p(0.001 + self.model.ASLM_acquisition_time / 1000)
        goingdownpoints = self.model.ao.s2p(0.001 + self.model.ASLM_acquisition_time / 1000 + 0.02) - goinguppoints
        sawtooth_array[0:goinguppoints] = np.linspace(ASLM_from, ASLM_to, goinguppoints)
        sawtooth_array[goinguppoints:] = np.linspace(ASLM_to, ASLM_from, goingdownpoints)

        basic_unit[:, NI_board_parameters.voicecoil] = self.model.smooth_sawtooth(sawtooth_array,
                                                                                window_len=self.model.ao.s2p(0.01))

        return basic_unit

    def get_lowRes_StackAq_array(self, current_laserline):
        # the camera needs time to read out the pixels - this is the camera readout time, and it adds to the
        # exposure time, depending on the number of rows that are imaged
        #nb_rows = 2960
        nb_rows = self.model.current_lowresROI_height
        readout_time = (nb_rows+1) * Camera_parameters.lowres_line_digitization_time #+1 for the reset time at the first row before the start

        # prepare voltage array
        # calculate minimal unit duration and set up array
        minimal_trigger_timeinterval = self.model.exposure_time_LR / 1000 + readout_time / 1000 + self.model.delay_cameratrigger + 0.002
        basic_unit = np.zeros((self.model.ao.s2p(minimal_trigger_timeinterval), NI_board_parameters.ao_nchannels),
                              np.dtype(np.float64))

        # set voltages in array - camera, stage, remote mirror, laser
        basic_unit[self.model.ao.s2p(self.model.delay_cameratrigger):self.model.ao.s2p(self.model.delay_cameratrigger + 0.002),
        NI_board_parameters.lowres_camera] = 4.  # low res camera
        basic_unit[0:self.model.ao.s2p(0.002), NI_board_parameters.stage] = 4.  # stage

        if current_laserline > 0:  # no-laser for LED
            basic_unit[self.model.ao.s2p(self.model.delay_cameratrigger):self.model.ao.s2p(self.model.delay_cameratrigger+self.model.exposure_time_LR / 1000),
            current_laserline] = 4.  # laser
        basic_unit[:, NI_board_parameters.voicecoil] = self.model.ASLM_staticLowResVolt  # remote mirror

        return basic_unit

    def get_highResSPIM_StackAq_array(self, current_laserline):
        # the camera needs time to read out the pixels - this is the camera readout time, and it adds to the
        # exposure time, depending on the number of rows that are imaged
        nb_rows = 2480
        nb_rows = self.model.current_highresROI_height

        readout_time = (nb_rows+1) * Camera_parameters.highres_line_digitization_time #+1 for the reset time at the first row before the start


        # prepare voltage array
        # calculate minimal unit duration and set up array
        minimal_trigger_timeinterval = self.model.exposure_time_HR / 1000 + readout_time / 1000 + self.model.delay_cameratrigger + 0.001
        basic_unit = np.zeros((self.model.ao.s2p(minimal_trigger_timeinterval), NI_board_parameters.ao_nchannels),
                              np.dtype(np.float64))

        # set voltages in array - camera, stage, remote mirror, laser
        basic_unit[self.model.ao.s2p(self.model.delay_cameratrigger):self.model.ao.s2p(self.model.delay_cameratrigger + 0.001),
        NI_board_parameters.highres_camera] = 4.  # highrescamera - ao0
        basic_unit[0:self.model.ao.s2p(0.002), NI_board_parameters.stage] = 4.  # stage
        basic_unit[:, NI_board_parameters.voicecoil] = self.model.ASLM_staticHighResVolt  # remote mirror
        if current_laserline > 0:  # no-laser for LED
            basic_unit[self.model.ao.s2p(self.model.delay_cameratrigger):self.model.ao.s2p(self.model.delay_cameratrigger+self.model.exposure_time_HR / 1000),
            current_laserline] = 4.  # laser

        return basic_unit

    def get_highResASLM_StackAq_array(self, current_laserline):

        #nb_rows = 2480: maximal number
        nb_rows = self.model.current_highresROI_height
        readout_time = (nb_rows+1) * Camera_parameters.highres_line_digitization_time * self.model.ASLM_line_delay #+1 for the reset time at the first row before the start

        # prepare voltage array
        # calculate minimal unit duration and set up array
        minimal_trigger_timeinterval = self.model.ASLM_acquisition_time / 1000 + \
                                       readout_time / 1000 +\
                                       self.model.delay_cameratrigger +\
                                       self.model.ASLM_delaybeforevoltagereturn + \
                                       self.model.ASLM_additionalreturntime +\
                                       0.001 #camera acquisition starts after camera puls

        basic_unit = np.zeros((self.model.ao.s2p(minimal_trigger_timeinterval), NI_board_parameters.ao_nchannels),
                              np.dtype(np.float64))

        # set voltages in array - camera, stage, laser
        basic_unit[self.model.ao.s2p(self.model.delay_cameratrigger):self.model.ao.s2p(self.model.delay_cameratrigger + 0.001),
        NI_board_parameters.highres_camera] = 4 #camera trigger to start ASLM acquisition
        basic_unit[0:self.model.ao.s2p(0.002), NI_board_parameters.stage] = 4.  # stage - move to position

        if current_laserline > 0:  # no-laser for LED
            basic_unit[
            self.model.ao.s2p(self.model.delay_cameratrigger):self.model.ao.s2p(self.model.delay_cameratrigger + self.model.ASLM_acquisition_time / 1000),
            current_laserline] = 4.  # laser


        ASLM_from = self.model.ASLM_from_Volt[self.model.current_laser - NI_board_parameters.adjustmentfactor]
        ASLM_to = self.model.ASLM_to_Volt[self.model.current_laser - NI_board_parameters.adjustmentfactor]

        # remote mirror voltage
        sawtooth_array = np.zeros(self.model.ao.s2p(minimal_trigger_timeinterval), np.dtype(np.float64))
        sawtooth_array[:] = ASLM_from
        goinguppoints = self.model.ao.s2p(
            self.model.delay_cameratrigger + 0.001 + self.model.ASLM_delaybeforevoltagereturn + self.model.ASLM_acquisition_time / 1000)
        goingdownpoints = self.model.ao.s2p(minimal_trigger_timeinterval) - goinguppoints
        sawtooth_array[0:goinguppoints] = np.linspace(ASLM_from, ASLM_to, goinguppoints)
        sawtooth_array[goinguppoints:] = np.linspace(ASLM_to, ASLM_from, goingdownpoints)

        basic_unit[:, NI_board_parameters.voicecoil] = sawtooth_array  # remote mirror

        return basic_unit

