
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class Stages_Tab(tk.Frame):
    """
    A stages tab to select which positions will be imaged in a timelapse
    - table to display selected positions
    - activate keyboard for movement and add positions (a,s,w,d and r,t)
    - change speed of stages for selecting
    - a tool to make a mosaic of the selected positions

    """

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # intro-text
        intro_text = tk.Label(self, text='In this tab, select the positions to image \n', height=2, width=115, fg="black", bg="grey")
        intro_text.grid(row=0, column=0, columnspan= 5000, sticky=(tk.E))

        # general stage settings
        self.stage_trans_stepsize = tk.DoubleVar()
        self.linear_stage_trans_stepsize = tk.DoubleVar()
        self.stage_rot_stepsize = tk.DoubleVar()

        # parameters move to
        self.stage_moveto_lateral = tk.DoubleVar()
        self.stage_moveto_axial   = tk.DoubleVar()
        self.stage_moveto_updown  = tk.DoubleVar()
        self.stage_moveto_angle   = tk.DoubleVar()

        # parameters save to position
        self.stage_currentPosindex = tk.IntVar()
        self.stage_currentPosindex.set(1)
        self.stage_currenthighresPosindex = tk.IntVar()
        self.stage_currenthighresPosindex.set(1)
        self.stage_PositionList = [(1,0,0,0,0)]
        self.stage_savePositionList = [(1,0,0,0,0)]
        self.stage_oldPositionList = [(1,0,0,0,0)]
        self.stage_highres_PositionList = [(1, 0, 0, 0, 0, 1)]
        self.stage_highres_savePositionList = [(1, 0, 0, 0, 0, 0)]
        self.stage_highres_oldPositionList = [(1, 0, 0, 0, 0, 0)]
        self.stage_mosaic_upDown = 2
        self.stage_mosaic_lateral = 2
        self.stage_last_key = tk.StringVar()
        self.stage_move_to_specificposition = tk.IntVar()

        #mosaic parameters
        self.stage_mosaic_upDown_Nb = tk.IntVar()
        self.stage_mosaic_leftright_Nb = tk.IntVar()
        self.stage_mosaic_upDown_Spacing = tk.DoubleVar()
        self.stage_mosaic_leftright_Spacing = tk.DoubleVar()

        # set the different label frames
        generalstage_settings = tk.LabelFrame(self, text="Stage Movement Settings")
        movetoposition = tk.LabelFrame(self, text="Move to ...")
        mosaic_settings = tk.LabelFrame(self, text="Mosaic settings")
        saved_lowRes_positions = tk.LabelFrame(self, text="Low Resolution Positions")
        saved_highres_positions = tk.LabelFrame(self, text="High Resolution Positions")

        # overall positioning of label frames
        generalstage_settings.grid(row=1, column=0, rowspan=2, sticky=tk.W + tk.E + tk.S + tk.N)
        movetoposition.grid(row=3, column=0, rowspan=17, sticky=tk.W + tk.E + tk.S + tk.N)
        mosaic_settings.grid(row=20, column=0, sticky=tk.W + tk.E + tk.S + tk.N)
        saved_lowRes_positions.grid(row=1, column=1, rowspan=10, sticky=tk.W + tk.E + tk.S + tk.N)
        saved_highres_positions.grid(row=12, column=1, rowspan=10, columnspan=2000, sticky=tk.W + tk.E + tk.S + tk.N)

        ### ----------------------------general stage settings -----------------------------------------------------------------
        # stage labels (positioned)
        stagestepsizelabel = ttk.Label(generalstage_settings, text="Trans. stage step size:").grid(row=0, column=0)
        anglestepsizelabel = ttk.Label(generalstage_settings, text="Rot. stage step size:").grid(row=5, column=0)
        mmstepsizelabel = ttk.Label(generalstage_settings, text="mm").grid(row=2, column=4)
        anglestepsizelabel = ttk.Label(generalstage_settings, text="degree").grid(row=8, column=4)
        rotstagestepsizelabel = ttk.Label(generalstage_settings, text="Rot. stage step size:").grid(row=6, column=0)

        transstage_scale = tk.Scale(generalstage_settings, variable=self.linear_stage_trans_stepsize,from_=0, to=2, resolution = 0.001, orient="horizontal", showvalue=False, command=self.update_stage_trans_stepsize)
        self.stage_trans_entry = tk.Entry(generalstage_settings, textvariable=self.stage_trans_stepsize, width=7)
        self.stage_rot_entry = tk.Entry(generalstage_settings, textvariable=self.stage_rot_stepsize, width=7)
        rotstage_scale = tk.Scale(generalstage_settings, variable=self.stage_rot_stepsize, from_=0, to=360,
                                  resolution=0.1, orient="horizontal")
        #default values
        self.linear_stage_trans_stepsize.set(2.00)
        self.stage_trans_stepsize.set(2.000)
        self.stage_rot_stepsize.set(2.000)

        #general stage settings widgets layout
        self.stage_trans_entry.grid(row=3, column=4, sticky=tk.W + tk.E)
        transstage_scale.grid(row=2, column=0, rowspan =2, sticky=tk.W + tk.E)
        rotstage_scale.grid(row=6, column=0, rowspan=2, sticky=tk.W + tk.E)
        self.stage_rot_entry.grid(row=7, column=4, sticky=tk.W + tk.E)



        ### ----------------------------move to position -----------------------------------------------------------------
        # move to labels (positioned)
        position_label = ttk.Label(movetoposition, text="Position").grid(row=0, column=1)
        positionX_label = ttk.Label(movetoposition, text="X").grid(row=2, column=0)
        positionY_label = ttk.Label(movetoposition, text="Y").grid(row=4, column=0)
        positionZ_label = ttk.Label(movetoposition, text="Z").grid(row=6, column=0)
        positionAngle_label = ttk.Label(movetoposition, text="Phi").grid(row=8, column=0)
        movetospecificPosition_label = ttk.Label(movetoposition, text="Move to position:").grid(row=14, column=0, columnspan=2)

        self.stage_move_left_bt = tk.Button(movetoposition, text="<", command=lambda : self.change_currentposition(self.stage_moveto_lateral, -1))
        self.stage_move_right_bt = tk.Button(movetoposition, text=">", command=lambda : self.change_currentposition(self.stage_moveto_lateral, 1))
        self.stage_move_up_bt = tk.Button(movetoposition, text="/\ ", command=lambda : self.change_currentposition(self.stage_moveto_updown, 1))
        self.stage_move_down_bt = tk.Button(movetoposition, text="\/", command=lambda : self.change_currentposition(self.stage_moveto_updown, -1))
        self.stage_move_forwardAxial_bt = tk.Button(movetoposition, text="Z-", command=lambda : self.change_currentposition(self.stage_moveto_axial, -1))
        self.stage_move_backwardAxial_bt = tk.Button(movetoposition, text="Z+", command=lambda : self.change_currentposition(self.stage_moveto_axial, 1))
        self.stage_move_angleleft_bt = tk.Button(movetoposition, text="R-", command=lambda : self.change_angle(self.stage_moveto_angle, -1))
        self.stage_move_angleright_bt = tk.Button(movetoposition, text="R+", command=lambda : self.change_angle(self.stage_moveto_angle, 1))

        self.stage_moveto_lateral_entry = tk.Entry(movetoposition, textvariable=self.stage_moveto_lateral, width=7)
        self.stage_moveto_updown_entry = tk.Entry(movetoposition, textvariable=self.stage_moveto_updown, width=7)
        self.stage_moveto_axial_entry = tk.Entry(movetoposition, textvariable=self.stage_moveto_axial, width=7)
        self.stage_moveto_angle_entry = tk.Entry(movetoposition, textvariable=self.stage_moveto_angle, width=7)

        self.keyboardinput = tk.StringVar(value="off")
        self.keyboard_input_on_bt = tk.Radiobutton(movetoposition, text="Enable Keyboard", value="on", variable =self.keyboardinput, indicatoron=False)
        self.keyboard_input_off_bt = tk.Radiobutton(movetoposition, text="Disable Keyboard", value="off", variable =self.keyboardinput, indicatoron=False)
        self.keyboard_entry = tk.Entry(movetoposition, textvariable=self.stage_last_key, width=7)

        self.move_to_specificPosition_Entry = tk.Entry(movetoposition, textvariable=self.stage_move_to_specificposition, width=7)
        self.move_to_specificPosition_Button = tk.Button(movetoposition, text="Move")
        self.move_to_specific_pos_resolution = tk.StringVar(value="on")
        self.move_to_specific_pos_low_on = tk.Radiobutton(movetoposition, text="Low", value="on",
                                                   variable=self.move_to_specific_pos_resolution, indicatoron=False)
        self.move_to_specific_pos_low_off = tk.Radiobutton(movetoposition, text="High", value="off",
                                                    variable=self.move_to_specific_pos_resolution, indicatoron=False)

        # move to widgets layout
        self.stage_moveto_lateral_entry.grid(row=2, column=1,columnspan=1,sticky = tk.W + tk.E)
        self.stage_moveto_updown_entry.grid(row=4, column=1,columnspan=1,sticky = tk.W + tk.E)
        self.stage_moveto_axial_entry.grid(row=6, column=1,columnspan=1,sticky = tk.W + tk.E)
        self.stage_moveto_angle_entry.grid(row=8, column=1,columnspan=1,sticky = tk.W + tk.E)

        self.stage_move_left_bt.grid(row=2, column=3,columnspan=1,sticky = tk.W + tk.E)
        self.stage_move_right_bt.grid(row=2, column=5,columnspan=1,sticky = tk.W + tk.E)
        self.stage_move_up_bt.grid(row=4, column=3,columnspan=1,sticky = tk.W + tk.E)
        self.stage_move_down_bt.grid(row=4, column=5,columnspan=1,sticky = tk.W + tk.E)
        self.stage_move_forwardAxial_bt.grid(row=6, column=3,columnspan=1,sticky = tk.W + tk.E)
        self.stage_move_backwardAxial_bt.grid(row=6, column=5,columnspan=1,sticky = tk.W + tk.E)
        self.stage_move_angleleft_bt.grid(row=8, column=3,columnspan=1,sticky = tk.W + tk.E)
        self.stage_move_angleright_bt.grid(row=8, column=5,columnspan=1,sticky = tk.W + tk.E)
        self.keyboard_input_on_bt.grid(row=12, column=0,columnspan=2,sticky = tk.W + tk.E)
        self.keyboard_input_off_bt.grid(row=12, column=2,columnspan=4,sticky = tk.W + tk.E)
        self.keyboard_entry.grid(row=12, column=6,columnspan=2,sticky = tk.W + tk.E)

        self.move_to_specificPosition_Entry.grid(row=15, column=0, columnspan=2, sticky=tk.W + tk.E)
        self.move_to_specificPosition_Button.grid(row=15, column=7, columnspan=2, sticky=tk.W + tk.E)
        self.move_to_specific_pos_low_on.grid(row=15, column=3, columnspan=2, sticky=tk.W + tk.E)
        self.move_to_specific_pos_low_off.grid(row=15, column=5, columnspan=2, sticky=tk.W + tk.E)

        ### ----------------------------mosaic settings -----------------------------------------------------------------
        # stage labels (positioned)
        mosaic_nbupdownlabel = ttk.Label(mosaic_settings, text="Up-Down #:").grid(row=0, column=0)
        mosaic_leftrightlabel = ttk.Label(mosaic_settings, text="Left-Right #:").grid(row=2, column=0)
        mosaic_stepsizeupdownlabel = ttk.Label(mosaic_settings, text="Up-Down Spacing:").grid(row=4, column=0)
        mosaic_leftrightupdownlabel = ttk.Label(mosaic_settings, text="Left-Right Spacing:").grid(row=6, column=0)

        self.stage_mosaic_updownNb_entry = tk.Entry(mosaic_settings, textvariable=self.stage_mosaic_upDown_Nb, width=7)
        self.stage_mosaic_leftrightNb_entry = tk.Entry(mosaic_settings, textvariable=self.stage_mosaic_leftright_Nb, width=7)
        self.stage_mosaic_updownSpa_entry = tk.Entry(mosaic_settings, textvariable=self.stage_mosaic_upDown_Spacing, width=7)
        self.stage_mosaic_leftrightSpa_entry = tk.Entry(mosaic_settings, textvariable=self.stage_mosaic_leftright_Spacing, width=7)

        self.stage_make_lowresMosaic_bt = tk.Button(mosaic_settings, text="Make Low Res Mosaic", command=lambda : self.makeMosaic("lowres"))
        self.stage_make_highresMosaic_bt = tk.Button(mosaic_settings, text="Make High Res Mosaic", command=lambda : self.makeMosaic("highres"))


        # default values
        self.stage_mosaic_upDown_Nb.set(2)
        self.stage_mosaic_leftright_Nb.set(1)
        self.stage_mosaic_upDown_Spacing.set(0.500)
        self.stage_mosaic_leftright_Spacing.set(0.500)

        # general stage settings widgets layout
        self.stage_mosaic_updownNb_entry.grid(row=0, column=3, sticky=tk.W + tk.E)
        self.stage_mosaic_leftrightNb_entry.grid(row=2, column=3, sticky=tk.W + tk.E)
        self.stage_mosaic_updownSpa_entry.grid(row=4, column=3, sticky=tk.W + tk.E)
        self.stage_mosaic_leftrightSpa_entry.grid(row=6, column=3, sticky=tk.W + tk.E)
        self.stage_make_lowresMosaic_bt.grid(row=8, column=0, sticky=tk.W + tk.E)
        self.stage_make_highresMosaic_bt.grid(row=8, column=3, sticky=tk.W + tk.E)
        ### ----------------------------low resolution saved positions -----------------------------------------------------------------
        # labels (positioned)
        position_label_lowres = ttk.Label(saved_lowRes_positions, text="Position:").grid(row=0, column=0)
        self.stage_savedPos_tree = ttk.Treeview(saved_lowRes_positions, columns=("Position", "X", "Y", "Z", "Phi"), show="headings", height=9)
        self.stage_addPos_bt = tk.Button(saved_lowRes_positions, text="Add position", command=lambda : self.addPos())
        self.stage_deletePos_bt = tk.Button(saved_lowRes_positions, text="Delete position", command=lambda : self.deletePos())
        self.stage_savePos_bt = tk.Button(saved_lowRes_positions, text="Save list", command=lambda : self.savePosList())
        self.stage_loadPos_bt = tk.Button(saved_lowRes_positions, text="Load saved list", command=lambda : self.loadPosList())
        self.stage_Revert_bt = tk.Button(saved_lowRes_positions, text="Revert", command=lambda : self.revertList())
        self.stage_addPos_index_entry = tk.Entry(saved_lowRes_positions, textvariable=self.stage_currentPosindex, width=4)

        ybarSrolling = tk.Scrollbar(saved_lowRes_positions, orient =tk.VERTICAL, command=self.stage_savedPos_tree.yview())
        self.stage_savedPos_tree.configure(yscroll=ybarSrolling.set)

        self.stage_savedPos_tree.heading("Position", text="Position")
        self.stage_savedPos_tree.heading("X", text="X")
        self.stage_savedPos_tree.heading("Y", text="Y")
        self.stage_savedPos_tree.heading("Z", text="Z")
        self.stage_savedPos_tree.heading("Phi", text="Angle")
        self.stage_savedPos_tree.column("Position", minwidth=0, width=55, stretch="NO", anchor="center")
        self.stage_savedPos_tree.column("X", minwidth=0, width=100, stretch="NO", anchor="center")
        self.stage_savedPos_tree.column("Y", minwidth=0, width=100, stretch="NO", anchor="center")
        self.stage_savedPos_tree.column("Z", minwidth=0, width=100, stretch="NO", anchor="center")
        self.stage_savedPos_tree.column("Phi", minwidth=0, width=100, stretch="NO", anchor="center")

        # Add content using (where index is the position/row of the treeview)
        # iid is the item index (used to access a specific element in the treeview)
        # you can set iid to be equal to the index
        tuples = [(1,0,0,0,0)]
        index = iid = 1
        for row in tuples:
            self.stage_savedPos_tree.insert("", 1, iid='item1', values=row)
            index = iid = index + 1


        # saved position layout
        self.stage_addPos_bt.grid(row=0,column=2,sticky = tk.W)
        self.stage_addPos_index_entry.grid(row=0,column=1,sticky = tk.W)
        self.stage_deletePos_bt.grid(row=0,column=3,sticky = tk.W)
        self.stage_savePos_bt.grid(row=0,column=4,sticky = tk.W)
        self.stage_loadPos_bt.grid(row=0,column=5,sticky = tk.W)
        self.stage_Revert_bt.grid(row=0,column=6,sticky = tk.W)
        self.stage_savedPos_tree.grid(row=2, column=0, columnspan=400)

        ### ----------------------------high resolution saved positions -----------------------------------------------------------------
        # labels (positioned)
        position_label_highres = ttk.Label(saved_lowRes_positions, text="Position:").grid(row=0, column=0)
        self.stage_highres_addPos_bt = tk.Button(saved_highres_positions, text="Add position",
                                         command=lambda: self.addhighresPos())
        self.stage_highres_deletePos_bt = tk.Button(saved_highres_positions, text="Delete position", command=lambda : self.deletehighresPos())
        self.stage_highres_savePos_bt = tk.Button(saved_highres_positions, text="Save list", command=lambda: self.savehighresPosList())
        self.stage_highres_loadPos_bt = tk.Button(saved_highres_positions, text="Load saved list",
                                          command=lambda: self.loadhighresPosList())
        self.stage_highres_Revert_bt = tk.Button(saved_highres_positions, text="Revert", command=lambda: self.reverthighresList())
        self.stage_highres_addPos_index_entry = tk.Entry(saved_highres_positions, textvariable=self.stage_currenthighresPosindex,
                                                 width=4)
        self.stage_highres_savedPos_tree = ttk.Treeview(saved_highres_positions, columns=("Position", "X", "Y", "Z", "Phi", "Label"),
                                                show="headings", height=9)

        ybarSrolling = tk.Scrollbar(saved_highres_positions, orient=tk.VERTICAL,
                                    command=self.stage_highres_savedPos_tree.yview())
        self.stage_highres_savedPos_tree.configure(yscroll=ybarSrolling.set)

        self.stage_highres_savedPos_tree.heading("Position", text="Position")
        self.stage_highres_savedPos_tree.heading("X", text="X")
        self.stage_highres_savedPos_tree.heading("Y", text="Y")
        self.stage_highres_savedPos_tree.heading("Z", text="Z")
        self.stage_highres_savedPos_tree.heading("Phi", text="Angle")
        self.stage_highres_savedPos_tree.heading("Label", text="Label")

        self.stage_highres_savedPos_tree.column("Position", minwidth=0, width=55, stretch="NO", anchor="center")
        self.stage_highres_savedPos_tree.column("X", minwidth=0, width=100, stretch="NO", anchor="center")
        self.stage_highres_savedPos_tree.column("Y", minwidth=0, width=100, stretch="NO", anchor="center")
        self.stage_highres_savedPos_tree.column("Z", minwidth=0, width=100, stretch="NO", anchor="center")
        self.stage_highres_savedPos_tree.column("Phi", minwidth=0, width=80, stretch="NO", anchor="center")
        self.stage_highres_savedPos_tree.column("Label", minwidth=0, width=55, stretch="NO", anchor="center")

        # Add content using (where index is the position/row of the treeview)
        # iid is the item index (used to access a specific element in the treeview)
        # you can set iid to be equal to the index
        tuples_high = [(1, 0, 0, 0, 0, 1)]
        index = iid = 1
        for row in tuples_high:
            self.stage_highres_savedPos_tree.insert("", 1, iid='item1', values=row)
            index = iid = index + 1

        # saved position layout
        self.stage_highres_addPos_bt.grid(row=0, column=2, sticky=tk.W)
        self.stage_highres_addPos_index_entry.grid(row=0, column=1, sticky=tk.W)
        self.stage_highres_deletePos_bt.grid(row=0,column=3,sticky=tk.W)
        self.stage_highres_savePos_bt.grid(row=0, column=4, sticky=tk.W)
        self.stage_highres_loadPos_bt.grid(row=0, column=5, sticky=tk.W)
        self.stage_highres_Revert_bt.grid(row=0, column=6, sticky=tk.W)
        self.stage_highres_savedPos_tree.grid(row=2, column=0, columnspan=500)
    #-------functions---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

    def update_stage_trans_stepsize(self, event):
        newvalue = 100 * 0.02 ** (3-self.linear_stage_trans_stepsize.get())
        newvalue = round(newvalue,5)
        self.stage_trans_stepsize.set(newvalue)

    def change_currentposition(self, direction, factor):
        new_position = round(direction.get() + self.stage_trans_stepsize.get() * factor,7)
        direction.set(new_position)

    def change_angle(self, direction, factor):
        new_position = round(direction.get() + self.stage_rot_stepsize.get() * factor, 5)

        if new_position < 0:
            new_position = 360 + new_position
        if new_position > 360:
            new_position = new_position-360

        direction.set(new_position)

    def savePosList(self):
        self.stage_savePositionList = self.stage_PositionList.copy()

    def savehighresPosList(self):
        self.stage_highres_savePositionList = self.stage_highres_PositionList.copy()

    def deletePos(self):
        # save previous state
        self.stage_oldPositionList = self.stage_PositionList.copy()

        #find and remove element from list
        for listiter in range(len(self.stage_PositionList)):
            print(self.stage_PositionList[listiter][0])
            if self.stage_PositionList[listiter][0] == self.stage_currentPosindex.get():
                self.stage_PositionList.remove(self.stage_PositionList[listiter])
                break # you remove one element - break for loop as otherwise error message appears (out of range)

        #display new tree
        self.display_tree(self.stage_savedPos_tree, self.stage_PositionList)

    def deletehighresPos(self):
        # save previous state
        self.stage_highres_oldPositionList = self.stage_highres_PositionList.copy()

        # find and remove element from list
        for listiter in range(len(self.stage_highres_PositionList)):
            if self.stage_highres_PositionList[listiter][0] == self.stage_currenthighresPosindex.get():
                self.stage_highres_PositionList.remove(self.stage_highres_PositionList[listiter])
                break # you remove one element - break for loop as otherwise error message appears (out of range)

        # display new tree
        self.display_tree(self.stage_highres_savedPos_tree, self.stage_highres_PositionList)

    def loadPosList(self):
        # save previous state
        self.stage_oldPositionList = self.stage_PositionList.copy()
        #load list
        self.stage_PositionList = self.stage_savePositionList.copy()
        # display tree
        self.display_tree(self.stage_savedPos_tree, self.stage_PositionList)

    def loadhighresPosList(self):
        # save previous state
        self.stage_highres_oldPositionList = self.stage_highres_PositionList.copy()
        #load list
        self.stage_highres_PositionList = self.stage_highres_savePositionList.copy()
        # display tree
        self.display_tree(self.stage_highres_savedPos_tree, self.stage_highres_PositionList)

    def revertList(self):
        self.stage_PositionList = self.stage_oldPositionList.copy()
        # display tree
        self.display_tree(self.stage_savedPos_tree, self.stage_PositionList)

    def reverthighresList(self):
        self.stage_highres_PositionList = self.stage_highres_oldPositionList.copy()
        # display tree
        self.display_tree(self.stage_highres_savedPos_tree, self.stage_highres_PositionList)

    def addPos(self):
        #save previous state
        self.stage_oldPositionList = self.stage_PositionList.copy()

        #new position to add or update
        newentry = (self.stage_currentPosindex.get(), self.stage_moveto_lateral.get(), self.stage_moveto_updown.get(), self.stage_moveto_axial.get(), self.stage_moveto_angle.get())

        #check if element is already there
        modified =0
        for listiter in range(len(self.stage_PositionList)):
            if self.stage_PositionList[listiter][0] == self.stage_currentPosindex.get():
                print(self.stage_PositionList[listiter])
                modified=1
                self.stage_PositionList[listiter] = newentry

        if modified==0:
            self.stage_PositionList.append(newentry)

        #sort list
        self.stage_PositionList.sort()

        #display tree
        self.display_tree(self.stage_savedPos_tree, self.stage_PositionList)

    def addhighresPos(self):
        #save previous state
        self.stage_highres_oldPositionList = self.stage_highres_PositionList.copy()

        #new position to add or update
        newentry = (self.stage_currenthighresPosindex.get(), self.stage_moveto_lateral.get(), self.stage_moveto_updown.get(), self.stage_moveto_axial.get(), self.stage_moveto_angle.get(), self.stage_currenthighresPosindex.get())

        #check if element is already there
        modified =0
        for listiter in range(len(self.stage_highres_PositionList)):
            if self.stage_highres_PositionList[listiter][0] == self.stage_currenthighresPosindex.get():
                print(self.stage_highres_PositionList[listiter])
                modified=1
                self.stage_highres_PositionList[listiter] = newentry

        if modified==0:
            self.stage_highres_PositionList.append(newentry)

        #sort list
        self.stage_highres_PositionList.sort()

        #display tree
        self.display_tree(self.stage_highres_savedPos_tree, self.stage_highres_PositionList)

    def display_tree(self, tree, positionlist):
        #delete current tree
        tree.delete(*tree.get_children())

        #generate new tree for display
        iter =0
        for listelement in positionlist:
            iter=iter+1
            newitem =iter+1
            tree.insert("", index=iter, iid=newitem, values=listelement)

    def makeMosaic(self, camera):

        #backup and select which camera
        if camera == "highres":
            poslist = self.stage_highres_PositionList
            self.stage_highres_oldPositionList = self.stage_highres_PositionList.copy()
        else:
            poslist = self.stage_PositionList
            self.stage_oldPositionList = self.stage_PositionList.copy()

        #make mosaic
        newlist = []
        positioniter =1
        for listiter in range(len(poslist)):
            currentposition = poslist[listiter]
            for updown_iter in range(self.stage_mosaic_upDown_Nb.get()):
                for leftright_iter in range(self.stage_mosaic_leftright_Nb.get()):
                    newposition = (positioniter, currentposition[1] + leftright_iter * self.stage_mosaic_leftright_Spacing.get(), currentposition[2] + updown_iter * self.stage_mosaic_upDown_Spacing.get(), currentposition[3], currentposition[4])
                    newlist.append(newposition)
                    positioniter = positioniter+1

        # display tree
        if camera == "highres":
            self.stage_highres_PositionList = newlist
            self.display_tree(self.stage_highres_savedPos_tree, self.stage_highres_PositionList)
            self.update_idletasks()
        else:
            self.stage_PositionList = newlist
            self.display_tree(self.stage_savedPos_tree, self.stage_PositionList)
            self.update_idletasks()
