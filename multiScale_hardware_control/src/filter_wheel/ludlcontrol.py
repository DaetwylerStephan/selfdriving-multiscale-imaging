
import serial as Serial
import io as Io
import time


class LudlFilterwheel:

    """ Class to control a 6-position Ludl filterwheel

    Needs a dictionary which combines filter designations and position IDs
    in the form:

    filters = {'405-488-647-Tripleblock' : 0,
           '405-488-561-640-Quadrupleblock': 1,
           '464 482-35': 2,
           '508 520-35': 3,
           '515LP':4,
           '529 542-27':5,
           '561LP':6,
           '594LP':7,
           'Empty-Alignment':8,}

    If there are tuples instead of integers as values, the
    filterwheel is assumed to be a double wheel.

    I.e.: '508 520-35': (2,3)

    adapted from ludl filter wheel control by Fabian Voigt
    """

    def __init__(self, COMport, filterdict, baudrate=9600):
        print('Initializing Ludl Filter Wheel')
        self.COMport = COMport
        self.baudrate = baudrate
        self.filterdict = filterdict
        self.double_wheel = False

        ''' Delay in s for the wait until done function '''
        self.wait_until_done_delay = 0.5

        """
        If the first entry of the filterdict has a tuple
        as value, it is assumed that it is a double-filterwheel
        to change the serial commands accordingly.

        TODO: This doesn't check that the tuple has length 2.
        """
        self.first_item_in_filterdict = list(self.filterdict.keys())[0]

        if type(self.filterdict[self.first_item_in_filterdict]) is tuple:
            self.double_wheel = True

    def close(self):
        '''
        Close the filter wheel
        '''

    def _check_if_filter_in_filterdict(self, filter):
        '''
        Checks if the filter designation (string) given as argument
        exists in the filterdict
        '''
        if filter in self.filterdict:
            return True
        else:
            raise ValueError('Filter designation not in the configuration')

    def set_filter(self, filter, wait_until_done=False):
        '''
        Moves filter using the pyserial command set.

        No checks are done whether the movement is completed or
        finished in time.


        '''
        if self._check_if_filter_in_filterdict(filter) is True:
            self.ser = Serial.Serial(self.COMport,
                                     self.baudrate,
                                     parity=Serial.PARITY_NONE,
                                     timeout=0,
                                     xonxoff=False,
                                     stopbits=Serial.STOPBITS_TWO)
            self.sio = Io.TextIOWrapper(Io.BufferedRWPair(self.ser, self.ser))
            """
            Check for double or single wheel

            TODO: A bit of repeating code in here. Might be better to
            spin the create and send commands off.
            """
            if self.double_wheel is False:
                """ Single wheel code """
                # Get the filter position from the filterdict:
                self.filternumber = self.filterdict[filter]
                # Rotat is the Ludl high-level command for moving a filter wheel
                self.ludlstring = 'Rotat S M ' + str(self.filternumber) + '\n'
                self.sio.write(str(self.ludlstring))
                self.sio.flush()
                self.ser.close()

                if wait_until_done:
                    ''' Wait a certain number of seconds. This is a hack

                    Testing with :
                    self.sio.write(str('Rdstat S'))
                    self.sio.flush()
                    print('First:', self.sio.readline(10))
                    time.sleep(0.1)
                    self.sio.write(str('Rdstat S'))
                    self.sio.flush()
                    print('Second: ', self.sio.readline(10))

                    yielded very unstable results, sometimes ":N -3", sometimes
                    ":A" - and blocking & crashing the connection
                    '''
                    time.sleep(self.wait_until_done_delay)

            else:
                """ Double wheel code """
                # Get the filter position tuple from the filterdict:
                self.filternumber = self.filterdict[filter]
                """ Write command for the primary wheel """
                self.ludlstring0 = 'Rotat S M ' + str(self.filternumber[0]) + '\n'
                self.sio.write(str(self.ludlstring0))
                """ Write command for the auxillary wheel """
                self.ludlstring1 = 'Rotat S A ' + str(self.filternumber[1]) + '\n'
                self.sio.write(str(self.ludlstring1))
                self.sio.flush()
                self.ser.close()

                if wait_until_done:
                    time.sleep(self.wait_until_done_delay)
        else:
            print(f'Filter {filter} not found in configuration.')


if __name__ == '__main__':
    ##test here code of this class
    ComPort = 'COM6'
    filters = {'515-30-25': 0,
               '572/20-25': 1,
               '615/20-25': 2,
               '676/37-25': 3,
               'transmission': 4,
               'block': 5,
               }
    filterwheel_test = LudlFilterwheel(ComPort, filters)
    filterwheel_test.set_filter('515-30-25', wait_until_done=False)
    filterwheel_test.set_filter('572/20-25', wait_until_done=False)
    filterwheel_test.set_filter('615/20-25', wait_until_done=False)
    filterwheel_test.set_filter('676/37-25', wait_until_done=False)