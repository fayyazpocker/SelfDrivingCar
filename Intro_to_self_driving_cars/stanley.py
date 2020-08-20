import numpy as np

class Stanley(object):
    def __init__(self, config):
        self.k = float(config.get('k', 0.1)) # crosstrack error gain
        self.L = 3 # Length of the car
    
    def map2veh(self, x_y_map, xf, yf, yaw):
        '''
        Converts the waypoints which is in reference with the space frame to that of vehicle

        x_y_map : x and y coordinates of the waypoints wrt space frame[[x1,y1],...]
        xf : x position of the center of the front axle wrt space frame
        yf : y position of the center of the front axle wrt space frame
        yaw : yaw of the vehicle wrt space frame (Left hand convention i.e x is right and y is downward)

        x_y_veh : x and y coordinates of the waypoints wrt front axle of the car
        '''
        rows = x_y_map.shape[0]
        x_y_veh = np.zeros([rows,2])
        
        # Position of the car wrt space frame. Keep in mind that the coordinate system in carla is of Left hand convention
        Tsc = np.array([[np.cos(yaw), -np.sin(yaw), 0, xf],[np.sin(yaw), np.cos(yaw),0,yf],[0,0,1,0],[0,0,0,1]])

        # Finding position of each waypoint wrt to the car
        for ind,wp in enumerate(x_y_map):    
            Tsw = np.array([[1,0,0,wp[0]],[0,1,0,wp[1]],[0,0,1,0],[0,0,0,1]]) # waypoint wrt space frame
            Tcw = np.linalg.inv(Tsc).dot(Tsw) # Tcw = Tsc^-1 * Tsw
            x_y_veh[ind,0] = Tcw[0,-1]
            x_y_veh[ind,1] = Tcw[1,-1]

        return x_y_veh



    def compute(self,waypoints, x, y, yaw, v):
        '''
        Computes the steering angle according to stanley controller

        waypoints : x and y coordinates of the waypoints wrt space frame[[x1,y1],...]
        x : x position of the center of the car wrt space frame
        y : y position of the center of the car space frame
        yaw : yaw of the vehicle wrt space frame (Left hand convention i.e x axis is right and y axis is downward)
        '''
        
        wps = np.squeeze(waypoints)

        # Finding front axle coordinates 
        xf = x + np.cos(yaw)*self.L/2
        yf = y - np.sin(yaw)*self.L/2

        # Converting into vehicle perpective
        wps_veh = self.map2veh(wps[:,:2], xf, yf, yaw)

        # Fitting the waypoints to a 1D line i.e ax + by +c form and finding a,b and c
        curve_coeff = np.polyfit(wps_veh[:,0], wps_veh[:,1], 1)
        yfit = np.poly1d(curve_coeff)

        a = yfit[1]
        b = -1
        c = yfit[0]

        # crosstack error = |axf + byf + c|/sqrt(a**2 + b**2). But as we already converted into 
        # car's coordinate frame, xf and yf is
        cte = c/np.sqrt(a**2 + b**2) 

        cte_steer = np.arctan(self.k * cte / (v+0.00001)) # steering is proportional to crosstrack error
        head_error = np.arctan(-a/b) # head error is tan inverse of the slope of the line as (0,0) of the line is the center of the front axle


        steering_angle =  head_error + cte_steer # Stanley controller

        # print (cte, cte_steer)
        # print (head_error, np.arctan2(-a,b), yaw, steering_angle)
        # print ("\n\n")
        # print ("a ", a)
        # print ("b ", b)
        # print ("c ", c)
        # print ("cte ", cte)
        # print ("cte_steer ", cte_steer)
        # # print ("yaw ", yaw)
        # print ("np.arctan(-a/b) ", np.arctan(-a/b))
        # print ("head_error ", head_error)
        # print ("steering angle ", steering_angle)

        return steering_angle



