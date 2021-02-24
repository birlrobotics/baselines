import math
from math import pi,cos,sin,acos

import numpy as np
from numpy.linalg import inv

from gym.envs.robotics import rotations as r_tool
#from ipdb import set_trace

class ger_learning:
    def __init__(self,env_name=None,err_distance = 0.05):
        self.err_distance = err_distance
        self.env_name = env_name

    def process_goals(self,goals):
        ''' Take current goals and at some x,y,z within the radius of the sphere
        '''
        goals_len = goals.shape[0]
        xs,ys,zs = self.generate_random_point_in_sphere(goals_len)
        
        # Add noise to goals
        for i,(offset_x,offset_y,offset_z) in enumerate(zip(xs,ys,zs)):
            goals[i] = goals[i] + np.array([offset_x,offset_y,offset_z])

        # Check for table limits
        if  self.env_name == "FetchSlide-v1" or self.env_name == "BaxterSlide-v1":
            # Y-coordinates
            for aug_goal in goals:
                if aug_goal[1]> 1.2:    # TODO: change to automate these coordinates according to the environment.
                    aug_goal[1] = 1.2
                elif aug_goal[1]<0.3:
                    aug_goal[1] = 0.3

            # X-coordinates
                if aug_goal[0] > 1.945:
                    aug_goal[0] = 1.945
                elif aug_goal[0] < 0.695:
                    aug_goal[0] = 0.695
        return goals.copy()
        
    def generate_random_point_in_sphere(self,goals_len): 
        ''' Parametric Equation of a Sphere (x,y,z)=(ρcosθsinϕ,ρsinθsinϕ,ρcosϕ), 
        where ρ is the constant radius, θ∈[0,2π) is the longitude and ϕ∈[0,π] is the colatitude.
        '''

        angle1s         = np.random.random(size=goals_len)*2*pi # Get random angles around the circle.
        random_radians  = np.random.random(size=goals_len)*2-1  # 2-1?

        angle2s = []
        for rand_rad in random_radians:
            angle2 = acos(rand_rad)
            angle2s.append(angle2)

        angle2s = np.asarray(angle2s, dtype=np.float32)
        rs=np.random.random(size=goals_len)**(1/3) # why (1/3)

        xs = []
        ys = []
        zs = []
        
        if self.env_name == "FetchSlide-v1" or self.env_name == "BaxterSlide-v1":
            for a1,a2,r in zip(angle1s,angle2s,rs):
                x=r*cos(a1)*sin(a2) * self.err_distance
                y=r*sin(a1)*sin(a2) * self.err_distance
                z= 0

                xs.append(x)
                ys.append(y)
                zs.append(z)

        elif self.env_name == "FetchPickAndPlace-v1" or self.env_name == "FetchPush-v1" or self.env_name == "BaxterPickAndPlace-v1" or self.env_name == "BaxterPush-v1":
            for a1,a2,r in zip(angle1s,angle2s,rs):
                x=r*cos(a1)*sin(a2) * self.err_distance
                y=r*sin(a1)*sin(a2) * self.err_distance
                z=r*cos(a2) * self.err_distance

                xs.append(x)
                ys.append(y)
                zs.append(z)
        else:
            assert("No such env :",self.env_name)
        return xs,ys,zs