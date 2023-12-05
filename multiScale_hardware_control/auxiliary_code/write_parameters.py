


class write_Params:
    def __init__(self, view):
        self.view = view


    def write_to_textfile(self, filePath, roilowres, roihighres):
        with open(filePath, 'w') as f:
            f.write('Experiment parameters of ' + filePath + "\n")
            f.write('---------------------------------------------\n')
            f.write('Plane spacing lowres: ' + str(self.view.runtab.stack_aq_plane_spacing_lowres.get()) + "\n")
            f.write('Plane spacing highres: ' + str(self.view.runtab.stack_aq_plane_spacing_highres.get()) + "\n")
            f.write('Nb of planes lowres: ' + str(self.view.runtab.stack_aq_numberOfPlanes_lowres.get()) + "\n")
            f.write('Nb of planes highres: ' + str(self.view.runtab.stack_aq_numberOfPlanes_highres.get()) + "\n")

            f.write('---------------------------------------------\n')
            f.write('ROI settings\n')
            f.write('roi lowres: ' + str(roilowres) + "\n")
            f.write('roi highres: ' + str(roihighres) + "\n")

            f.write('---------------------------------------------\n')
            f.write('Laser power\n')
            if self.view.runtab.stack_aq_488onLowRes.get() ==1:
                f.write('Laser power 488 lowres: ' + str(self.view.runtab.laser488_percentage_LR.get()) + "\n")
            if self.view.runtab.stack_aq_488onHighRes.get() == 1:
                f.write('Laser power 488 highres: ' + str(self.view.runtab.laser488_percentage_HR.get()) + "\n")
            if self.view.runtab.stack_aq_552onLowRes.get() == 1:
                f.write('Laser power 552 lowres: ' + str(self.view.runtab.laser552_percentage_LR.get()) + "\n")
            if self.view.runtab.stack_aq_552onHighRes.get() == 1:
                f.write('Laser power 552 highres: ' + str(self.view.runtab.laser552_percentage_HR.get()) + "\n")
            if self.view.runtab.stack_aq_594onLowRes.get() ==1:
                f.write('Laser power 594 lowres: ' + str(self.view.runtab.laser594_percentage_LR.get()) + "\n")
            if self.view.runtab.stack_aq_594onHighRes.get() ==1:
                f.write('Laser power 594 highres: ' + str(self.view.runtab.laser594_percentage_HR.get()) + "\n")
            if self.view.runtab.stack_aq_640onLowRes.get() ==1:
                f.write('Laser power 640 lowres: ' + str(self.view.runtab.laser640_percentage_LR.get()) + "\n")
            if self.view.runtab.stack_aq_640onHighRes.get() == 1:
                f.write('Laser power 640 highres: ' + str(self.view.runtab.laser640_percentage_HR.get()) + "\n")

            f.write('---------------------------------------------\n')
            f.write('exposure time settings\n')

            f.write('Low res exposure time: ' + str(self.view.runtab.cam_lowresExposure.get()) + "\n")
            f.write('High res exposure time: ' + str(self.view.runtab.cam_highresExposure.get()) + "\n")

            f.write('---------------------------------------------\n')
            f.write('low resolution stack positions\n')

            for iter_lowrespos in self.view.stagessettingstab.stage_savedPos_tree.get_children():
                #get current position from list
                xpos = float(self.view.stagessettingstab.stage_savedPos_tree.item(iter_lowrespos)['values'][1])
                ypos = float(self.view.stagessettingstab.stage_savedPos_tree.item(iter_lowrespos)['values'][2])
                zpos = float(self.view.stagessettingstab.stage_savedPos_tree.item(iter_lowrespos)['values'][3])
                angle = float(self.view.stagessettingstab.stage_savedPos_tree.item(iter_lowrespos)['values'][4])
                current_startposition = [xpos, ypos, zpos, angle]
                f.write(str(current_startposition) + "\n")

            f.write('\n---------------------------------------------\n')
            f.write('high resolution stack positions\n')

            for iter_highrespos in self.view.stagessettingstab.stage_highres_savedPos_tree.get_children():
                #get current position from list
                xpos = float(self.view.stagessettingstab.stage_highres_savedPos_tree.item(iter_highrespos)['values'][1])
                ypos = float(self.view.stagessettingstab.stage_highres_savedPos_tree.item(iter_highrespos)['values'][2])
                zpos = float(self.view.stagessettingstab.stage_highres_savedPos_tree.item(iter_highrespos)['values'][3])
                angle = float(self.view.stagessettingstab.stage_highres_savedPos_tree.item(iter_highrespos)['values'][4])
                current_startposition = [xpos, ypos, zpos, angle]
                f.write(str(current_startposition) + "\n")

            f.write('\n---------------------------------------------\n')
            f.write('SPIM / ASLM: ')
            if self.view.runtab.cam_highresMode.get() == "SPIM Mode":
                f.write("SPIM mode\n")
            else:
                f.write("ASLM mode\n")
                f.write('Volt interval (mV): ' + str(self.view.advancedSettingstab.ASLM_volt_interval.get()) + "\n")
                f.write('Volt center @488 (mV): ' + str(self.view.advancedSettingstab.ASLM_volt_middle488.get()) + "\n")
                f.write('Volt center @552 (mV): ' + str(self.view.advancedSettingstab.ASLM_volt_middle552.get()) + "\n")
                f.write('Volt center @594 (mV): ' + str(self.view.advancedSettingstab.ASLM_volt_middle594.get()) + "\n")
                f.write('Volt center @640 (mV): ' + str(self.view.advancedSettingstab.ASLM_volt_middle640.get()) + "\n")
                f.write('Orientation: ' + str(self.view.advancedSettingstab.ASLM_voltageDirection.get()) + "\n")
                f.write('Scan width: ' + str(self.view.advancedSettingstab.ASLM_scanWidth.get()) + "\n")

            f.write('\n---------------------------------------------\n')
            f.write('time-lapse settings: ')

            totaltime = self.view.runtab.timelapse_aq_timeinterval_min.get() * 60 + self.view.runtab.timelapse_aq_timeinterval_seconds.get()
            f.write('time interval: ' + str(totaltime) + "\n")