from ctypes import *
import time
import os
import sys
import platform
import tempfile
import re


class slit_ximc_control:

    def __init__(self):
        '''
        Initialize xilinc slit, and print out some parameters for debugging
        Input:
        Output: an initialized and connected slit.
        '''

        print("a")
        print(sys.version_info)

        if sys.version_info >= (3, 0):
            import urllib.parse

        # Dependences

        # For correct usage of the library libximc, add the file pyximc.py wrapper
        # with the structures of the library to python path.
        self.cur_dir = os.path.abspath(os.path.dirname(__file__))  # Specifies the current directory.
        print(self.cur_dir)
        self.ximc_dir = os.path.join(self.cur_dir, "ximc")  # Formation of the directory name with all dependencies. The dependencies for the examples are located in the ximc directory.
        print(self.ximc_dir)
        self.ximc_package_dir = os.path.join(self.ximc_dir, "crossplatform", "wrappers",
                                "python")  # Formation of the directory name with python dependencies.
        print(self.ximc_package_dir)
        sys.path.append(self.ximc_package_dir)  # add pyximc.py wrapper to python path

        # Depending on your version of Windows, add the path to the required DLLs to the environment variable
        # bindy.dll
        # libximc.dll
        # xiwrapper.dll
        if platform.system() == "Windows":
            # Determining the directory with dependencies for windows depending on the bit depth.
            arch_dir = "win64" if "64" in platform.architecture()[0] else "win32"  #
            self.libdir = os.path.join(self.ximc_dir, arch_dir)
            os.environ["Path"] = self.libdir + ";" + os.environ["Path"]  # add dll path into an environment variable

        try:
            import pyximc
        except ImportError as err:
            print(
                "Can't import pyximc module. The most probable reason is that you changed the relative location of the testpython.py and pyximc.py files. See developers' documentation for details.")
            exit()
        except OSError as err:
            # print(err.errno, err.filename, err.strerror, err.winerror) # Allows you to display detailed information by mistake.
            if platform.system() == "Windows":
                if err.winerror == 193:  # The bit depth of one of the libraries bindy.dll, libximc.dll, xiwrapper.dll does not correspond to the operating system bit.
                    print(
                        "Err: The bit depth of one of the libraries bindy.dll, libximc.dll, xiwrapper.dll does not correspond to the operating system bit.")
                    # print(err)
                elif err.winerror == 126:  # One of the library bindy.dll, libximc.dll, xiwrapper.dll files is missing.
                    print("Err: One of the library bindy.dll, libximc.dll, xiwrapper.dll is missing.")
                    # print(err)
                else:  # Other errors the value of which can be viewed in the code.
                    print(err)
                print(
                    "Warning: If you are using the example as the basis for your module, make sure that the dependencies installed in the dependencies section of the example match your directory structure.")
                print("For correct work with the library you need: pyximc.py, bindy.dll, libximc.dll, xiwrapper.dll")
            else:
                print(err)
                print(
                    "Can't load libximc library. Please add all shared libraries to the appropriate places. It is decribed in detail in developers' documentation. On Linux make sure you installed libximc-dev package.\nmake sure that the architecture of the system and the interpreter is the same")
            exit()

        # variable 'lib' points to a loaded library
        # note that ximc uses stdcall on win
        print("Library loaded")

        self.lib = pyximc.lib
        self.x_device_information = pyximc.device_information_t()
        self.Result = pyximc.Result
        self.x_status = pyximc.status_t()
        self.x_pos = pyximc.get_position_t()
        self.eng = pyximc.engine_settings_t()
        self.MicrostepMode = pyximc.MicrostepMode
        self.mvst = pyximc.move_settings_t()


        self.sbuf = create_string_buffer(64)
        self.lib.ximc_version(self.sbuf)
        print("Library version: " + self.sbuf.raw.decode().rstrip("\0"))

        # Set bindy (network) keyfile. Must be called before any call to "enumerate_devices" or "open_device" if you
        # wish to use network-attached controllers. Accepts both absolute and relative paths, relative paths are resolved
        # relative to the process working directory. If you do not need network devices then "set_bindy_key" is optional.
        # In Python make sure to pass byte-array object to this function (b"string literal").
        self.lib.set_bindy_key(os.path.join(self.ximc_dir, "win32", "keyfile.sqlite").encode("utf-8"))

        # This is device search and enumeration with probing. It gives more information about devices.
        probe_flags = pyximc.EnumerateFlags.ENUMERATE_PROBE + pyximc.EnumerateFlags.ENUMERATE_NETWORK
        enum_hints = b"addr=192.168.0.1,172.16.2.3"
        # enum_hints = b"addr=" # Use this hint string for broadcast enumerate
        self.devenum = self.lib.enumerate_devices(probe_flags, enum_hints)
        print("Device enum handle: " + repr(self.devenum))
        print("Device enum handle type: " + repr(type(self.devenum)))

        self.dev_count = self.lib.get_device_count(self.devenum)
        print("Device count: " + repr(self.dev_count))

        self.controller_name = pyximc.controller_name_t()
        for dev_ind in range(0, self.dev_count):
            enum_name = self.lib.get_device_name(self.devenum, dev_ind)
            result = self.lib.get_enumerate_device_controller_name(self.devenum, dev_ind, byref(self.controller_name))
            if result == self.Result.Ok:
                print("Enumerated device #{} name (port name): ".format(dev_ind) + repr(
                    enum_name) + ". Friendly name: " + repr(
                    self.controller_name.ControllerName) + ".")

        self.open_name = None
        if len(sys.argv) > 1:
            self.open_name = sys.argv[1]
        elif self.dev_count > 0:
            self.open_name = self.lib.get_device_name(self.devenum, 0)
        elif sys.version_info >= (3, 0):
            # use URI for virtual device when there is new urllib python3 API
            tempdir = tempfile.gettempdir() + "/testdevice.bin"
            if os.altsep:
                tempdir = tempdir.replace(os.sep, os.altsep)
            # urlparse build wrong path if scheme is not file
            uri = urllib.parse.urlunparse(urllib.parse.ParseResult(scheme="file", \
                                                                   netloc=None, path=tempdir, params=None, query=None,
                                                                   fragment=None))
            self.open_name = re.sub(r'^file', 'xi-emu', uri).encode()

        if not self.open_name:
            exit(1)

        if type(self.open_name) is str:
            self.open_name = self.open_name.encode()

        print("\nOpen device " + repr(self.open_name))
        self.device_id = self.lib.open_device(self.open_name)
        print("Device id: " + repr(self.device_id))




    def slit_info(self):
        print("\nGet device info")
        result = self.lib.get_device_information(self.device_id, byref(self.x_device_information))
        print("Result: " + repr(result))
        if result == self.Result.Ok:
            print("Device information:")
            print(" Manufacturer: " +
                repr(string_at(self.x_device_information.Manufacturer).decode()))
            print(" ManufacturerId: " +
                repr(string_at(self.x_device_information.ManufacturerId).decode()))
            print(" ProductDescription: " +
                repr(string_at(self.x_device_information.ProductDescription).decode()))
            print(" Major: " + repr(self.x_device_information.Major))
            print(" Minor: " + repr(self.x_device_information.Minor))
            print(" Release: " + repr(self.x_device_information.Release))


    def slit_status(self):
        print("\nGet status")
        result = self.lib.get_status(self.device_id, byref(self.x_status))
        print("Result: " + repr(result))
        if result == self.Result.Ok:
            print("Status.Ipwr: " + repr(self.x_status.Ipwr))
            print("Status.Upwr: " + repr(self.x_status.Upwr))
            print("Status.Iusb: " + repr(self.x_status.Iusb))
            print("Status.Flags: " + repr(hex(self.x_status.Flags)))


    def slit_get_position(self):
        print("\nRead position")
        result = self.lib.get_position(self.device_id, byref(self.x_pos))
        print("Result: " + repr(result))
        if result == self.Result.Ok:
            print("Position: {0} steps, {1} microsteps".format(self.x_pos.Position, self.x_pos.uPosition))
        return self.x_pos.Position, self.x_pos.uPosition


    def slit_left(self):
        print("\nMoving left")
        result = self.lib.command_left(self.device_id)
        print("Result: " + repr(result))


    def slit_move(self, distance, udistance):
        print("\nGoing to {0} steps, {1} microsteps".format(distance, udistance))
        result = self.lib.command_move(self.device_id, distance, udistance)
        print("Result: " + repr(result))

    def slit_wait_for_stop(self, interval):
        print("\nWaiting for stop")
        result = self.lib.command_wait_for_stop(self.device_id, interval)
        print("Result: " + repr(result))


    def slit_serial(self):
        print("\nReading serial")
        x_serial = c_uint()
        result = self.lib.get_serial_number(self.device_id, byref(x_serial))
        if result == self.Result.Ok:
            print("Serial: " + repr(x_serial.value))


    def slit_get_speed(self):
        print("\nGet speed")
        # Create move settings structure
        # Get current move settings from controller
        result = self.lib.get_move_settings(self.device_id, byref(self.mvst))
        # Print command return status. It will be 0 if all is OK
        print("Read command result: " + repr(result))

        return self.mvst.Speed


    def slit_set_speed(self, speed):
        print("\nSet speed")
        # Create move settings structure
        # Get current move settings from controller
        result = self.lib.get_move_settings(self.device_id, byref(self.mvst))
        # Print command return status. It will be 0 if all is OK
        print("Read command result: " + repr(result))
        print("The speed was equal to {0}. We will change it to {1}".format(self.mvst.Speed, speed))
        # Change current speed
        self.mvst.Speed = int(speed)
        # Write new move settings to controller
        result = self.lib.set_move_settings(self.device_id, byref(self.mvst))
        # Print command return status. It will be 0 if all is OK
        print("Write command result: " + repr(result))


    def slit_set_microstep_mode_256(self):
        print("\nSet microstep mode to 1/256 steps")
        # Create engine settings structure
        # Get current engine settings from controller
        result = self.lib.get_engine_settings(self.device_id, byref(self.eng))
        # Print command return status. It will be 0 if all is OK
        print("Read command result: " + repr(result))
        # Change MicrostepMode parameter to MICROSTEP_MODE_FRAC_256
        # (use MICROSTEP_MODE_FRAC_128, MICROSTEP_MODE_FRAC_64 ... for other microstep modes)
        self.eng.MicrostepMode = self.MicrostepMode.MICROSTEP_MODE_FRAC_256
        # Write new engine settings to controller
        result = self.lib.set_engine_settings(self.device_id, byref(self.eng))
        # Print command return status. It will be 0 if all is OK
        print("Write command result: " + repr(result))

    def slit_set_microstep_mode_2(self):
        print("\nSet microstep mode to 1/2 steps")
        # Create engine settings structure
        # Get current engine settings from controller
        result = self.lib.get_engine_settings(self.device_id, byref(self.eng))
        # Print command return status. It will be 0 if all is OK
        print("Read command result: " + repr(result))
        # Change MicrostepMode parameter to MICROSTEP_MODE_FRAC_256
        # (use MICROSTEP_MODE_FRAC_128, MICROSTEP_MODE_FRAC_64 ... for other microstep modes)
        self.eng.MicrostepMode = self.MicrostepMode.MICROSTEP_MODE_FRAC_2
        # Write new engine settings to controller
        result = self.lib.set_engine_settings(self.device_id, byref(self.eng))
        # Print command return status. It will be 0 if all is OK
        print("Write command result: " + repr(result))

    def home_stage(self):
        print("\nSet Position to Zero")
        self.lib.command_homezero(self.device_id)

    def slit_closing(self):
        print("\nSlit Closing")
        self.lib.close_device(byref(cast(self.device_id, POINTER(c_int))))


if __name__ == '__main__':
    ##test here code of this class
    print("aaa")
    test_slit = slit_ximc_control()
    test_slit.slit_info()
    test_slit.slit_status()
    test_slit.slit_set_microstep_mode_256()
    print("----------------------------------------\n home\n-------------------------------------")
    test_slit.home_stage()

    startpos, ustartpos = test_slit.slit_get_position()

    print(startpos)
    print(ustartpos)

    # first move
    test_slit.slit_set_speed(500)
    #test_slit.slit_left()
    time.sleep(4)
    test_slit.slit_get_position()
    # second move
    current_speed = test_slit.slit_get_speed()
    test_slit.slit_set_speed(1800)

    test_slit.slit_move(startpos, ustartpos)
    test_slit.slit_move(4000, 0)
    test_slit.slit_move(4000, 0)
    test_slit.slit_wait_for_stop(100)
    test_slit.slit_status()
    test_slit.slit_serial()

    test_slit.slit_closing()

    print("Done")
