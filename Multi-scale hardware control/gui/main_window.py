"""A better Hello World for Tkinter"""

import tkinter as tk
from tkinter import ttk

try:
    from .Welcome_window import Welcome_Tab
    from .run_window import Run_Tab
    from .stages_window import Stages_Tab
    from .advancedsettings_window import AdvancedSettings_Tab
    from .smart_window import SmartMicroscopySettings_Tab

except ImportError:
    from Welcome_window import Welcome_Tab
    from run_window import Run_Tab
    from stages_window import Stages_Tab
    from advancedsettings_window import AdvancedSettings_Tab
    from smart_window import SmartMicroscopySettings_Tab




class MultiScope_MainGui(ttk.Notebook):
    """
    This is the main GUI class for the multi-scale microscope. It arranges the microscope GUI into different tabs:
    a welcome tab, a settings tab, a stage settings tab, a run tab and the advanced settings
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        # set the window properties

        #define the individual sheets: a welcome tab, a settings tab, a stage settings tab, a run tab and the advanced settings
        self.runtab = Run_Tab(self)
        self.welcometab = Welcome_Tab(self)
        #self.settingstab = Settings_Tab(self)
        self.stagessettingstab = Stages_Tab(self)
        self.advancedSettingstab = AdvancedSettings_Tab(self)
        self.automatedMicroscopySettingstab = SmartMicroscopySettings_Tab(self)

        #add the individual sheets to the Notebook
        self.add(self.welcometab, text = "Welcome")
        #self.add(self.settingstab, text="Settings")
        self.add(self.stagessettingstab, text="Stages")
        self.add(self.runtab, text="Run")
        self.add(self.advancedSettingstab, text="Advanced Settings")
        self.add(self.automatedMicroscopySettingstab, text="Smart Settings")

        #pack tge sheets
        self.pack(expand=1, fill='both')

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Multi-scale microscope V1")
    root.geometry("800x600")
    all_tabs_mainGUI = ttk.Notebook(root)
    Gui_mainwindow = MultiScope_MainGui(all_tabs_mainGUI)
    root.mainloop()
