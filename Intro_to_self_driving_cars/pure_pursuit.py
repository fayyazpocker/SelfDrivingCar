import numpy as np

class PurePursuit(object):
    def __init__(self, config):
        self.K_dd = float(config.get('Kdd', 0.1))
        self.min_ld = float(config.get('min_ld', 0.1))
        self.L = 3 # Length of the car
    
    def compute(self, v, x, y, yaw, waypoints):
        '''
        Computes the steering angle using pure pursuit controller

        v : current velocity
        x : current x position
        y : current y position
        yaw : current yaw (Left hand convention as carla is left hand convention based)
        waypoints: list of waypoints [[x1,y1],[x2,y2],...]

        steering_angle : Steering angle required as per pure pursuit controller
        '''
        ld = max(self.K_dd * v, self.min_ld) # setting a minimum ld value
        wps = np.squeeze(waypoints)
        xr = x - (self.L * np.cos(yaw)/2) # Finding position of the rear axle from center of the car
        yr = y + (self.L * np.sin(yaw)/2)
        min_err = float('inf')
        ld_path = float('inf')

        # Traversing through waypoints to find distance almost same as that of the look ahead distance
        for wp in wps:
            dist = np.linalg.norm(np.array([wp[0] - xr, wp[1] - yr])) # Distance from rear axle to the waypoint
            if (abs(dist - ld) < min_err): # Saving the least error waypoint in the path equal to the fixed look ahead distance
                la_wp = wp
                min_err = abs(dist - ld)
                ld_path = dist
        
        alpha = np.arctan2((la_wp[1] - yr),(la_wp[0] - xr)) - yaw # reducing yaw from the slope to find alpha
        steering_angle = np.arctan2(2*self.L*np.sin(alpha), ld_path)

        # print (la_wp, ld_path, ld, alpha, steering_angle)
        return steering_angle

        


