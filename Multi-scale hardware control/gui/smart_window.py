
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


class SmartMicroscopySettings_Tab(tk.Frame):
    """
    A tab for advanced settings such as rotational stage calibration etc

    """

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # intro-text
        intro_text = tk.Label(self, text='In this tab, you have some settings available for automated microscopy \n', height=2,
                              width=115,
                              fg="black", bg="grey")
        intro_text.grid(row=0, column=0, columnspan=5000, sticky=(tk.E))

        #parameters
        self.drift_correction_highres = tk.IntVar()
        self.drift_correction_lowres = tk.IntVar()
        self.driftcorrection_488 = tk.IntVar()
        self.driftcorrection_552 = tk.IntVar()
        self.driftcorrection_594 = tk.IntVar()
        self.driftcorrection_640 = tk.IntVar()
        self.driftcorrection_LED = tk.IntVar()

        # set the different label frames
        drift_correction_settings = tk.LabelFrame(self, text="Drift Correction Option")
        drift_correction_settings.grid(row=1, column=3, rowspan=2, sticky=tk.W + tk.E + tk.S + tk.N)

        self.drift_highres_button = tk.Checkbutton(drift_correction_settings, text ='On High Resolution', variable=self.drift_correction_highres, onvalue=1, offvalue=0)
        self.drift_lowres_button = tk.Checkbutton(drift_correction_settings, text ='On Low Resolution', variable=self.drift_correction_lowres, onvalue=1, offvalue=0)

        self.drift_On488 = tk.Checkbutton(drift_correction_settings, text='488', variable=self.driftcorrection_488,
                                                  onvalue=1, offvalue=0)
        self.drift_On552 = tk.Checkbutton(drift_correction_settings, text='552', variable=self.driftcorrection_552,
                                                  onvalue=1, offvalue=0)
        self.drift_On594 = tk.Checkbutton(drift_correction_settings, text='594', variable=self.driftcorrection_594,
                                                  onvalue=1, offvalue=0)
        self.drift_On640 = tk.Checkbutton(drift_correction_settings, text='640', variable=self.driftcorrection_640,
                                                  onvalue=1, offvalue=0)
        self.drift_OnLED = tk.Checkbutton(drift_correction_settings, text='LED', variable=self.driftcorrection_LED,
                                                  onvalue=1, offvalue=0)



        #arrange
        self.drift_highres_button.grid(row=2, column=3, sticky=tk.W)
        self.drift_lowres_button.grid(row=3, column=3, sticky=tk.W)
        ttk.Label(drift_correction_settings, text="Which channel to correct:").grid(row=4, column=3, sticky=tk.W)
        self.drift_On488.grid(row=7, column=3)
        self.drift_On552.grid(row=8, column=3)
        self.drift_On594.grid(row=9, column=3)
        self.drift_On640.grid(row=10, column=3)
        self.drift_OnLED.grid(row=11, column=3)