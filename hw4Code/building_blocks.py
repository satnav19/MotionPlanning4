import math
import queue
import random
import time
import threading
import multiprocessing
import timeit

import numpy as np

import environment
from environment import LocationType

class Building_Blocks(object):
    '''
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    '''

    def __init__(self, env, resolution=0.1, p_bias=0.05, special_bias=False):
        self.env = env
        self.resolution = resolution
        self.p_bias = p_bias
        #TODO : ???????????????????????????
        #self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07])
        # testing variables
        self.t_curr = 0
        self.special_bias = special_bias

        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])

        self.single_mechanical_limit = list(self.ur_params.mechamical_limits.values())[-1][-1]

        # pairs of links that can collide during sampling
        self.possible_link_collisions = [['shoulder_link', 'forearm_link'],
                                         ['shoulder_link', 'wrist_1_link'],
                                         ['shoulder_link', 'wrist_2_link'],
                                         ['shoulder_link', 'wrist_3_link'],
                                         ['upper_arm_link', 'wrist_1_link'],
                                         ['upper_arm_link', 'wrist_2_link'],
                                         ['upper_arm_link', 'wrist_3_link'],
                                         ['forearm_link', 'wrist_2_link'],
                                         ['forearm_link', 'wrist_3_link']]

    def sample_random_config(self, goal_prob,  goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        # TODO: HW2 5.2.1
        if np.random.uniform(low=0, high=1) <= goal_prob:
            return goal_conf
        config = np.array([np.random.uniform(low=joint_limit[0], high=joint_limit[1]) 
                    for joint_limit in self.ur_params.mechamical_limits.values()])
        return config


    def config_validity_checker(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return True if in collision
        @param conf - some configuration
        """
        # TODO: HW2 5.2.2- Pay attention that function is a little different than in HW2
        coords = self.transform.conf2sphere_coords(conf)
        radii = self.ur_params.sphere_radius
        obstacle_rad = self.env.radius

        for joint,values in coords.items():
            for center in values:
                if center[0] + radii[joint] >=0.4: #wall in manipulator env, HW3 relevant
                    return False
                if center[0] !=0 and center[1] != 0 and center[2] < radii[joint]:
                    return False
                
                for obstacle in self.env.obstacles:
                    obstacle_dist = np.linalg.norm(center - obstacle)
                    rad_sum = radii[joint] + obstacle_rad
                    if  obstacle_dist < rad_sum:
                        return False
                    
        for collision in self.possible_link_collisions:
            joint1 = collision[0]
            joint2 = collision[1]
            rad_sum = radii[joint1] +  radii[joint2]
            joint1_centers = coords[joint1]
            joint2_centers = coords[joint2]

            for center1 in joint1_centers:
                for center2 in joint2_centers:
                    actual_distance = np.linalg.norm(center1 - center2)
                    if actual_distance < rad_sum:
                        return False
                            
        return True



    def edge_validity_checker(self, prev_conf, current_conf) -> bool:
        '''check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        '''
        # TODO: HW2 5.2.4
        if not self.config_validity_checker(current_conf):
            return False
        if not self.config_validity_checker(prev_conf):
            return False
        
        res = self.resolution
        max_joint_diff = np.max(np.abs(current_conf - prev_conf))
        num_steps = max(3, (int(max_joint_diff / res)+1))
        temp_conf = np.copy(prev_conf)
        for step in range(1, num_steps-1):
            t = step / (num_steps - 1)
            temp_conf = prev_conf + t * (current_conf - prev_conf)
            if not self.config_validity_checker(temp_conf):
                return False

        return True

    def edge_validity_checker_lazy(self, prev_conf, current_conf) -> bool:
        '''
        Quick preliminary collision check between two configurations
        Only checks endpoints and midpoint instead of full resolution
        @param prev_conf - some configuration
        @param current_conf - current configuration
        @return True if the transition might be valid (needs full check), False if definitely invalid
        '''
        if not self.config_validity_checker(current_conf):
            return False
        if not self.config_validity_checker(prev_conf):
            return False
    
        # Check midpoint
        temp_conf = np.copy(prev_conf)
        temp_conf += 0.5 * (current_conf - prev_conf)
        if not self.config_validity_checker(temp_conf):
            return False
    
        return True


    def compute_distance(self, conf1, conf2):
        '''
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        '''
        return np.dot(self.cost_weights, np.power(conf1 - conf2, 2)) ** 0.5
