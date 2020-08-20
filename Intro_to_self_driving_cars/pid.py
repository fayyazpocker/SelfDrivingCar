#!/usr/bin/env python3

import numpy as np

class PID(object):
    def __init__(self, config):

        self.sampletime = float(config.get('sampletime', 100)) # in ms
        Kp = float(config.get('Kp', 0))
        Kd = float(config.get('Kd', 0))
        Ki = float(config.get('Ki', 0))
        self.setTunings(Kp, Kd, Ki)

        self.min_out = float(config.get('min_out', 0))
        self.max_out = float(config.get('max_out', 1))

        self.ITerm = 0.0
        self.lastInput = 0.0
    
    def setTunings(self, Kp, Kd, Ki):
        '''
        Set tuning parameters
        '''
        sample_time_sec = float(self.sampletime/1000)
        self.Kp = Kp
        self.Kd = Kd/sample_time_sec
        self.Ki = Ki * sample_time_sec
    
    def setlimit(self, value):
        return np.fmax(np.fmin(value, self.max_out), self.min_out)
    
    def compute(self, input, setpoint):
        '''
        computing PID based on improved pid
        input : current input to the system
        setpoint : desired configuration

        output : PID output 
        '''
        error = setpoint - input
        self.ITerm += self.Ki * error # Integral term considering windup lag
        self.ITerm = self.setlimit(self.ITerm)
        dInput = input - self.lastInput # Derivative term considering derivative kick

        output = self.Kp * error + self.ITerm - self.Kd * dInput
        output = self.setlimit(output)

        self.lastInput = input

        return output

        
        