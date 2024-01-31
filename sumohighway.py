from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import optparse
from typing import List, Any, Union
import gym
from gym import spaces
import numpy as np
from numpy import float32


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci


optParser = optparse.OptionParser()
optParser.add_option("--nogui", action="store_true",
                     default=False, help="run the commandline version of sumo")
options, args = optParser.parse_args(args=[])
if options.nogui:
    sumoBinary = checkBinary('sumo')
else:
    sumoBinary = checkBinary('sumo-gui')

sumocfgfile = "C:\\Users\\PC\\Desktop\\algorithm\\highway\\sumo.sumocfg"  
traci.start([sumoBinary, "-c", sumocfgfile])

class HighwayEnv(gym.Env):
    def __init__(self):
        self.car_length = 5
        self.car_width = 1.8
        self.frame = 0
        self.state = None
        self.hard_brake = -4
        self.deceleration = -1.5
        self.acceleration = 1.5
        self.time = 0.5
        self.lateral_v = 1.33
        self.v_max_longitudinal = 33.3
        self.epoch = 0
        self.max_epoch = 5000
        self.action_space = spaces.Discrete(12)
        self.vehicle_position_xmax = 1200
        self.ego_vehicle_lanemax = 0
        self.ego_vehicle_longitudinalvmax = 40 
        self.ego_vehicle_lateralvmax = 1.5
        self.vehicle_position_xmin = -200
        self.ego_vehicle_lanemin = -16
        self.ego_vehicle_longitudinalvmin = 0
        self.ego_vehicle_lateralvmin = -1.5
        obs_high = np.array([self.vehicle_position_xmax, self.ego_vehicle_lanemax, self.ego_vehicle_longitudinalvmax, self.ego_vehicle_lateralvmax,
                            self.vehicle_position_xmax, self.ego_vehicle_lanemax, self.ego_vehicle_longitudinalvmax, self.ego_vehicle_lateralvmax,
                            self.vehicle_position_xmax, self.ego_vehicle_lanemax, self.ego_vehicle_longitudinalvmax, self.ego_vehicle_lateralvmax,
                            self.vehicle_position_xmax, self.ego_vehicle_lanemax, self.ego_vehicle_longitudinalvmax, self.ego_vehicle_lateralvmax,
                            self.vehicle_position_xmax, self.ego_vehicle_lanemax, self.ego_vehicle_longitudinalvmax, self.ego_vehicle_lateralvmax,
                            self.vehicle_position_xmax, self.ego_vehicle_lanemax, self.ego_vehicle_longitudinalvmax, self.ego_vehicle_lateralvmax,
                            self.vehicle_position_xmax, self.ego_vehicle_lanemax, self.ego_vehicle_longitudinalvmax, self.ego_vehicle_lateralvmax], dtype=np.float32)
        obs_low = np.array([self.vehicle_position_xmin, self.ego_vehicle_lanemin, self.ego_vehicle_longitudinalvmin, self.ego_vehicle_lateralvmin,
                             self.vehicle_position_xmin, self.ego_vehicle_lanemin, self.ego_vehicle_longitudinalvmin, self.ego_vehicle_lateralvmin,
                             self.vehicle_position_xmin, self.ego_vehicle_lanemin, self.ego_vehicle_longitudinalvmin, self.ego_vehicle_lateralvmin,
                             self.vehicle_position_xmin, self.ego_vehicle_lanemin, self.ego_vehicle_longitudinalvmin, self.ego_vehicle_lateralvmin,
                             self.vehicle_position_xmin, self.ego_vehicle_lanemin, self.ego_vehicle_longitudinalvmin, self.ego_vehicle_lateralvmin,
                             self.vehicle_position_xmin, self.ego_vehicle_lanemin, self.ego_vehicle_longitudinalvmin, self.ego_vehicle_lateralvmin,
                             self.vehicle_position_xmin, self.ego_vehicle_lanemin, self.ego_vehicle_longitudinalvmin, self.ego_vehicle_lateralvmin], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.lane_left = 0
        self.lane_right = 0

    def reset(self):
        self.frame = 1
        self.epoch += 1
        self.lane_left = 0
        self.lane_right = 0
        for i in range(100):
            traci.simulationStep()
        traci.simulationStep()  
        all_vehicle_id = traci.vehicle.getIDList()
        all_vehicle_position_x = [traci.vehicle.getPosition(i)[0] for i in all_vehicle_id]
        ego_vehicle_position = min(all_vehicle_position_x)
        ego_vehicle_i = all_vehicle_position_x.index(ego_vehicle_position)
        self.ego_vehicle_id = all_vehicle_id[ego_vehicle_i]
        traci.vehicle.setColor(self.ego_vehicle_id, (0, 0, 255))
        traci.vehicle.setLaneChangeMode(self.ego_vehicle_id, 0b000000000000)
        traci.vehicle.setSpeedMode(self.ego_vehicle_id, 0b000000) 
        
        ego_vehicle_position_x, ego_vehicle_lane = traci.vehicle.getPosition(self.ego_vehicle_id)
        ego_vehicle_longitudinalv = traci.vehicle.getSpeed(self.ego_vehicle_id)
        ego_vehicle_lateralv = traci.vehicle.getLateralSpeed(self.ego_vehicle_id)
        ego_front_position_x, ego_front_lane, ego_front_longitudinalv, ego_front_lateralv,\
        ego_rear_position_x, ego_rear_lane, ego_rear_longitudinalv, ego_rear_lateralv,\
        left_front_position_x, left_front_lane, left_front_longitudinalv, left_front_lateralv,\
        left_rear_position_x, left_rear_lane, left_rear_longitudinalv, left_rear_lateralv,\
        right_front_position_x, right_front_lane, right_front_longitudinalv, right_front_lateralv,\
        right_rear_position_x, right_rear_lane, right_rear_longitudinalv, right_rear_lateralv = self.current_state(ego_vehicle_position_x, ego_vehicle_lane)
        self.state = np.array([ego_vehicle_position_x, ego_vehicle_lane, ego_vehicle_longitudinalv, ego_vehicle_lateralv,
                               ego_front_position_x, ego_front_lane, ego_front_longitudinalv, ego_front_lateralv,
                               ego_rear_position_x, ego_rear_lane, ego_rear_longitudinalv, ego_rear_lateralv,
                               left_front_position_x, left_front_lane, left_front_longitudinalv, left_front_lateralv,
                               left_rear_position_x, left_rear_lane, left_rear_longitudinalv, left_rear_lateralv,
                               right_front_position_x, right_front_lane, right_front_longitudinalv, right_front_lateralv,
                               right_rear_position_x, right_rear_lane, right_rear_longitudinalv, right_rear_lateralv], dtype = float32)
        return self.normalization(self.state), self.state



    def step(self, action):
        global r_collision, r_speed
        self.frame += 1
        ego_vehicle_position_x, ego_vehicle_lane, ego_vehicle_longitudinalv, ego_vehicle_lateralv,\
        ego_front_position_x, ego_front_lane, ego_front_longitudinalv, ego_front_lateralv,\
        ego_rear_position_x, ego_rear_lane, ego_rear_longitudinalv, ego_rear_lateralv,\
        left_front_position_x, left_front_lane, left_front_longitudinalv, left_front_lateralv,\
        left_rear_position_x, left_rear_lane, left_rear_longitudinalv, left_rear_lateralv,\
        right_front_position_x, right_front_lane, right_front_longitudinalv, right_front_lateralv,\
        right_rear_position_x, right_rear_lane, right_rear_longitudinalv, right_rear_lateralv = self.state
        
        if action == 0:
            ego_vehicle_position_x = ego_vehicle_position_x + ego_vehicle_longitudinalv*self.time + 0.5*self.hard_brake*self.time**2
            ego_vehicle_longitudinalv = ego_vehicle_longitudinalv+self.hard_brake*self.time
            ego_vehicle_lateralv = self.lateral_v
            ego_vehicle_lane = ego_vehicle_lane
            self.lane_left += 1
            self.lane_right = 0
        elif action == 1:
            ego_vehicle_position_x = ego_vehicle_position_x + ego_vehicle_longitudinalv * self.time + 0.5 * self.hard_brake * self.time ** 2
            ego_vehicle_longitudinalv = ego_vehicle_longitudinalv + self.hard_brake * self.time
            ego_vehicle_lateralv = 0
            ego_vehicle_lane = ego_vehicle_lane
            self.lane_left = 0
            self.lane_right = 0
        elif action == 2:
            ego_vehicle_position_x = ego_vehicle_position_x + ego_vehicle_longitudinalv * self.time + 0.5 * self.hard_brake * self.time ** 2
            ego_vehicle_longitudinalv = ego_vehicle_longitudinalv + self.hard_brake * self.time
            ego_vehicle_lateralv = -self.lateral_v
            ego_vehicle_lane = ego_vehicle_lane
            self.lane_left = 0
            self.lane_right += 1
        elif action == 3:
            ego_vehicle_position_x = ego_vehicle_position_x + ego_vehicle_longitudinalv * self.time + 0.5 * self.deceleration * self.time ** 2
            ego_vehicle_longitudinalv = ego_vehicle_longitudinalv + self.deceleration * self.time
            ego_vehicle_lateralv = self.lateral_v
            ego_vehicle_lane = ego_vehicle_lane
            self.lane_left += 1
            self.lane_right = 0
        elif action == 4:
            ego_vehicle_position_x = ego_vehicle_position_x + ego_vehicle_longitudinalv * self.time + 0.5 * self.deceleration * self.time ** 2
            ego_vehicle_longitudinalv = ego_vehicle_longitudinalv + self.deceleration * self.time
            ego_vehicle_lateralv = 0
            ego_vehicle_lane = ego_vehicle_lane
            self.lane_left  = 0
            self.lane_right = 0
        elif action == 5:
            ego_vehicle_position_x = ego_vehicle_position_x + ego_vehicle_longitudinalv * self.time + 0.5 * self.deceleration * self.time ** 2
            ego_vehicle_longitudinalv = ego_vehicle_longitudinalv + self.deceleration * self.time
            ego_vehicle_lateralv = -self.lateral_v
            ego_vehicle_lane = ego_vehicle_lane
            self.lane_left = 0
            self.lane_right += 1
        elif action == 6:
            ego_vehicle_position_x = ego_vehicle_position_x + ego_vehicle_longitudinalv * self.time + 0.5 * self.acceleration * self.time ** 2
            ego_vehicle_longitudinalv = ego_vehicle_longitudinalv + self.acceleration * self.time
            ego_vehicle_lateralv = self.lateral_v
            ego_vehicle_lane = ego_vehicle_lane
            self.lane_left += 1
            self.lane_right = 0
        elif action == 7:
            ego_vehicle_longitudinalv = ego_vehicle_longitudinalv + self.acceleration * self.time
            ego_vehicle_lateralv = 0
            ego_vehicle_lane = ego_vehicle_lane
            self.lane_left = 0
            self.lane_right = 0
        elif action == 8:
            ego_vehicle_position_x = ego_vehicle_position_x + ego_vehicle_longitudinalv * self.time + 0.5 * self.acceleration * self.time ** 2
            ego_vehicle_longitudinalv = ego_vehicle_longitudinalv + self.acceleration * self.time
            ego_vehicle_lateralv = -self.lateral_v
            ego_vehicle_position_x = ego_vehicle_position_x + ego_vehicle_longitudinalv * self.time + 0.5 * self.acceleration * self.time ** 2
            ego_vehicle_lane = ego_vehicle_lane
            self.lane_left = 0
            self.lane_right += 1
        elif action == 9:
            ego_vehicle_position_x = ego_vehicle_position_x + ego_vehicle_longitudinalv * self.time
            ego_vehicle_longitudinalv = ego_vehicle_longitudinalv
            ego_vehicle_lateralv = self.lateral_v
            ego_vehicle_lane = ego_vehicle_lane
            self.lane_left += 1
            self.lane_right = 0
        elif action == 10:
            ego_vehicle_position_x = ego_vehicle_position_x + ego_vehicle_longitudinalv * self.time
            ego_vehicle_longitudinalv = ego_vehicle_longitudinalv
            ego_vehicle_lateralv = 0
            ego_vehicle_lane = ego_vehicle_lane
            self.lane_left = 0
            self.lane_right = 0
        elif action == 11:
            ego_vehicle_position_x = ego_vehicle_position_x + ego_vehicle_longitudinalv * self.time
            ego_vehicle_longitudinalv = ego_vehicle_longitudinalv
            ego_vehicle_lateralv = -self.lateral_v
            ego_vehicle_lane = ego_vehicle_lane
            self.lane_left = 0
            self.lane_right += 1
        

        ego_vehicle_longitudinalv = self.limit_speed(ego_vehicle_longitudinalv, ego_vehicle_lane)
        if self.lane_left == 6:
            if traci.vehicle.getLaneIndex(self.ego_vehicle_id) >= 3:
                traci.vehicle.changeLane(self.ego_vehicle_id,traci.vehicle.getLaneIndex(self.ego_vehicle_id),0.5)
                ego_vehicle_lane = ego_vehicle_lane
            else:
                traci.vehicle.changeLane(self.ego_vehicle_id,traci.vehicle.getLaneIndex(self.ego_vehicle_id)+1,0.5)
                ego_vehicle_lane = ego_vehicle_lane + 4
            self.lane_left = 0
            self.lane_right = 0
        elif self.lane_right == 6:
            if traci.vehicle.getLaneIndex(self.ego_vehicle_id) <= 0:
                traci.vehicle.changeLane(self.ego_vehicle_id,traci.vehicle.getLaneIndex(self.ego_vehicle_id),0.5)
                ego_vehicle_lane = ego_vehicle_lane
            else:
                traci.vehicle.changeLane(self.ego_vehicle_id,traci.vehicle.getLaneIndex(self.ego_vehicle_id)-1,0.5)
                ego_vehicle_lane = ego_vehicle_lane - 4
            self.lane_left = 0
            self.lane_right = 0
        traci.vehicle.setSpeed(self.ego_vehicle_id,ego_vehicle_longitudinalv)
        traci.simulationStep()  
        if traci.simulation.getCollidingVehiclesNumber() != 0:
            if self.ego_vehicle_id in traci.simulation.getCollidingVehiclesIDList():
                collision = True
                r_collision = -4
            else:
                collision = False
        else:
            r_collision = 0
            collision = False
            ego_vehicle_position_x = traci.vehicle.getPosition(self.ego_vehicle_id)[0]
            ego_vehicle_lane = traci.vehicle.getPosition(self.ego_vehicle_id)[1]
        ego_front_position_x, ego_front_lane, ego_front_longitudinalv, ego_front_lateralv,\
        ego_rear_position_x, ego_rear_lane, ego_rear_longitudinalv, ego_rear_lateralv,\
        left_front_position_x, left_front_lane, left_front_longitudinalv, left_front_lateralv,\
        left_rear_position_x, left_rear_lane, left_rear_longitudinalv, left_rear_lateralv,\
        right_front_position_x, right_front_lane, right_front_longitudinalv, right_front_lateralv,\
        right_rear_position_x, right_rear_lane, right_rear_longitudinalv, right_rear_lateralv = self.current_state(ego_vehicle_position_x, ego_vehicle_lane)
        self.state = np.array([ego_vehicle_position_x, ego_vehicle_lane, ego_vehicle_longitudinalv, ego_vehicle_lateralv,
                               ego_front_position_x, ego_front_lane, ego_front_longitudinalv, ego_front_lateralv,
                               ego_rear_position_x, ego_rear_lane, ego_rear_longitudinalv, ego_rear_lateralv,
                               left_front_position_x, left_front_lane, left_front_longitudinalv, left_front_lateralv,
                               left_rear_position_x, left_rear_lane, left_rear_longitudinalv, left_rear_lateralv,
                               right_front_position_x, right_front_lane, right_front_longitudinalv, right_front_lateralv,
                               right_rear_position_x, right_rear_lane, right_rear_longitudinalv, right_rear_lateralv], dtype=float32)
        done = bool(self.frame > 50 or collision == True)
        if self.frame > 50 and collision == False:
            traci.vehicle.remove(self.ego_vehicle_id)
       
        r_speed = 0.5*((ego_vehicle_longitudinalv-16.7)/(33.3-16.7))
        reward: float32
        reward = r_collision + r_speed
        return self.normalization(self.state), self.state, reward, done, collision



    def current_state(self, ego_vehicle_position_x, ego_vehicle_lane):

        global ego_front_position_x, ego_front_lane, ego_front_longitudinalv,ego_front_lateralv,ego_rear_position_x,ego_rear_lane,\
               ego_rear_longitudinalv, ego_rear_lateralv, left_front_position_x, left_front_lane, left_front_longitudinalv, left_front_lateralv,\
               left_rear_position_x, left_rear_lane, left_rear_longitudinalv, left_rear_lateralv, right_front_position_x, right_front_lane,\
               right_front_longitudinalv, right_front_lateralv, right_rear_position_x, right_rear_lane, right_rear_longitudinalv, right_rear_lateralv
        all_vehicle_id = traci.vehicle.getIDList()
        all_vehicle_position = [list((traci.vehicle.getPosition(i), i)) for i in all_vehicle_id]
        lane1 = list(filter(lambda x: x[0][1] == -2, all_vehicle_position))
        lane2 = list(filter(lambda x: x[0][1] == -6, all_vehicle_position))
        lane3 = list(filter(lambda x: x[0][1] == -10, all_vehicle_position))
        lane4 = list(filter(lambda x: x[0][1] == -14, all_vehicle_position))
        if -4 <= ego_vehicle_lane < 0:
            left_front_position_x, left_front_lane, left_front_longitudinalv, left_front_lateralv = ego_vehicle_position_x + 150, 2, 33.3, 0
            left_rear_position_x, left_rear_lane, left_rear_longitudinalv, left_rear_lateralv = ego_vehicle_position_x - 150, 2, 33.3, 0
            lane1_distance_front = list(filter(lambda x: 150 > x[0][0] - ego_vehicle_position_x > 0, lane1))  
            if not lane1_distance_front:
                ego_front_position_x, ego_front_lane, ego_front_longitudinalv, ego_front_lateralv = ego_vehicle_position_x + 150, -2, 33.3, 0
            else:
                lane1_distance_front_x = [lane1_distance_front[i][0][0] for i in range(len(lane1_distance_front))]
                lane1_distance_front_id = lane1_distance_front[lane1_distance_front_x.index(min(lane1_distance_front_x))][1]
                ego_front_position_x, ego_front_lane = traci.vehicle.getPosition(lane1_distance_front_id)
                ego_front_longitudinalv = traci.vehicle.getSpeed(lane1_distance_front_id)
                ego_front_lateralv = traci.vehicle.getLateralSpeed(lane1_distance_front_id)
            lane1_distance_rear = list(filter(lambda x: -150 < x[0][0] - ego_vehicle_position_x < 0, lane1))
            if not lane1_distance_rear:
                ego_rear_position_x, ego_rear_lane, ego_rear_longitudinalv, ego_rear_lateralv = ego_vehicle_position_x - 150, -2, 33.3, 0
            else:
                lane1_distance_rear_x = [lane1_distance_rear[i][0][0] for i in range(len(lane1_distance_rear))]
                lane1_distance_rear_id = lane1_distance_rear[lane1_distance_rear_x.index(max(lane1_distance_rear_x))][1]
                ego_rear_position_x, ego_rear_lane = traci.vehicle.getPosition(lane1_distance_rear_id)
                ego_rear_longitudinalv = traci.vehicle.getSpeed(lane1_distance_rear_id)
                ego_rear_lateralv = traci.vehicle.getLateralSpeed(lane1_distance_rear_id)
            lane2_distance_front = list(filter(lambda x: 150 > x[0][0] - ego_vehicle_position_x > 0, lane2))
            if not lane2_distance_front:
                right_front_position_x, right_front_lane, right_front_longitudinalv, right_front_lateralv = ego_vehicle_position_x + 150, -6, 33.3, 0
            else:
                lane2_distance_front_x = [lane2_distance_front[i][0][0] for i in range(len(lane2_distance_front))]
                lane2_distance_front_id = lane2_distance_front[lane2_distance_front_x.index(min(lane2_distance_front_x))][1]
                right_front_position_x, right_front_lane = traci.vehicle.getPosition(lane2_distance_front_id)
                right_front_longitudinalv = traci.vehicle.getSpeed(lane2_distance_front_id)
                right_front_lateralv = traci.vehicle.getLateralSpeed(lane2_distance_front_id)
            lane2_distance_rear = list(filter(lambda x: -150 < x[0][0] - ego_vehicle_position_x < 0, lane2))
            if not lane2_distance_rear:
                right_rear_position_x, right_rear_lane, right_rear_longitudinalv, right_rear_lateralv = ego_vehicle_position_x - 150, -6, 33.3, 0
            else:
                lane2_distance_rear_x = [lane2_distance_rear[i][0][0] for i in range(len(lane2_distance_rear))]
                lane2_distance_rear_id = lane2_distance_rear[lane2_distance_rear_x.index(max(lane2_distance_rear_x))][1]
                right_rear_position_x, right_rear_lane = traci.vehicle.getPosition(lane2_distance_rear_id)
                right_rear_longitudinalv = traci.vehicle.getSpeed(lane2_distance_rear_id)
                right_rear_lateralv = traci.vehicle.getLateralSpeed(lane2_distance_rear_id)
        elif -16 <= ego_vehicle_lane < -12:
            right_front_position_x, right_front_lane, right_front_longitudinalv, right_front_lateralv = ego_vehicle_position_x + 150, -18, 27.7, 0
            right_rear_position_x, right_rear_lane, right_rear_longitudinalv, right_rear_lateralv = ego_vehicle_position_x - 150, -18, 27.7, 0
            lane4_distance_front = list(filter(lambda x: 150 > x[0][0] - ego_vehicle_position_x > 0, lane4))  
            if not lane4_distance_front:
                ego_front_position_x, ego_front_lane, ego_front_longitudinalv, ego_front_lateralv = ego_vehicle_position_x + 150, -14, 27.7, 0
            else:
                lane4_distance_front_x = [lane4_distance_front[i][0][0] for i in range(len(lane4_distance_front))]
                lane4_distance_front_id = lane4_distance_front[lane4_distance_front_x.index(min(lane4_distance_front_x))][1]
                ego_front_position_x, ego_front_lane = traci.vehicle.getPosition(lane4_distance_front_id)
                ego_front_longitudinalv = traci.vehicle.getSpeed(lane4_distance_front_id)
                ego_front_lateralv = traci.vehicle.getLateralSpeed(lane4_distance_front_id)
            lane4_distance_rear = list(filter(lambda x: -150 < x[0][0] - ego_vehicle_position_x < 0, lane4))
            if not lane4_distance_rear:
                ego_rear_position_x, ego_rear_lane, ego_rear_longitudinalv, ego_rear_lateralv = ego_vehicle_position_x - 150, -14, 27.7, 0
            else:
                lane4_distance_rear_x = [lane4_distance_rear[i][0][0] for i in range(len(lane4_distance_rear))]
                lane4_distance_rear_id = lane4_distance_rear[lane4_distance_rear_x.index(max(lane4_distance_rear_x))][1]
                ego_rear_position_x, ego_rear_lane = traci.vehicle.getPosition(lane4_distance_rear_id)
                ego_rear_longitudinalv = traci.vehicle.getSpeed(lane4_distance_rear_id)
                ego_rear_lateralv = traci.vehicle.getLateralSpeed(lane4_distance_rear_id)
            lane3_distance_front = list(filter(lambda x: 150 > x[0][0] - ego_vehicle_position_x > 0, lane3))
            if not lane3_distance_front:
                left_front_position_x, left_front_lane, left_front_longitudinalv, left_front_lateralv = ego_vehicle_position_x + 150, -10, 27.7, 0
            else:
                lane3_distance_front_x = [lane3_distance_front[i][0][0] for i in range(len(lane3_distance_front))]
                lane3_distance_front_id = lane3_distance_front[lane3_distance_front_x.index(min(lane3_distance_front_x))][1]
                left_front_position_x, left_front_lane = traci.vehicle.getPosition(lane3_distance_front_id)
                left_front_longitudinalv = traci.vehicle.getSpeed(lane3_distance_front_id)
                left_front_lateralv = traci.vehicle.getLateralSpeed(lane3_distance_front_id)
            lane3_distance_rear = list(filter(lambda x: -150 < x[0][0] - ego_vehicle_position_x < 0, lane3))
            if not lane3_distance_rear:
                left_rear_position_x, left_rear_lane, left_rear_longitudinalv, left_rear_lateralv = ego_vehicle_position_x - 150, -10, 27.7, 0
            else:
                lane3_distance_rear_x = [lane3_distance_rear[i][0][0] for i in range(len(lane3_distance_rear))]
                lane3_distance_rear_id = lane3_distance_rear[lane3_distance_rear_x.index(max(lane3_distance_rear_x))][1]
                left_rear_position_x, left_rear_lane = traci.vehicle.getPosition(lane3_distance_rear_id)
                left_rear_longitudinalv = traci.vehicle.getSpeed(lane3_distance_rear_id)
                left_rear_lateralv = traci.vehicle.getLateralSpeed(lane3_distance_rear_id)
        elif -8 <= ego_vehicle_lane < -4:
            lane2_distance_front = list(filter(lambda x: 150 > x[0][0] - ego_vehicle_position_x > 0, lane2))  
            if not lane2_distance_front:
                ego_front_position_x, ego_front_lane, ego_front_longitudinalv, ego_front_lateralv = ego_vehicle_position_x + 150, -6, 33.3, 0
            else:
                lane2_distance_front_x = [lane2_distance_front[i][0][0] for i in range(len(lane2_distance_front))]
                lane2_distance_front_id = lane2_distance_front[lane2_distance_front_x.index(min(lane2_distance_front_x))][1]
                ego_front_position_x, ego_front_lane = traci.vehicle.getPosition(lane2_distance_front_id)
                ego_front_longitudinalv = traci.vehicle.getSpeed(lane2_distance_front_id)
                ego_front_lateralv = traci.vehicle.getLateralSpeed(lane2_distance_front_id)
            lane2_distance_rear = list(filter(lambda x: -150 < x[0][0] - ego_vehicle_position_x < 0, lane2))
            if not lane2_distance_rear:
                ego_rear_position_x, ego_rear_lane, ego_rear_longitudinalv, ego_rear_lateralv = ego_vehicle_position_x - 150, -6, 33.3, 0
            else:
                lane2_distance_rear_x = [lane2_distance_rear[i][0][0] for i in range(len(lane2_distance_rear))]
                lane2_distance_rear_id = lane2_distance_rear[lane2_distance_rear_x.index(max(lane2_distance_rear_x))][1]
                ego_rear_position_x, ego_rear_lane = traci.vehicle.getPosition(lane2_distance_rear_id)
                ego_rear_longitudinalv = traci.vehicle.getSpeed(lane2_distance_rear_id)
                ego_rear_lateralv = traci.vehicle.getLateralSpeed(lane2_distance_rear_id)
            lane1_distance_front = list(filter(lambda x: 150 > x[0][0] - ego_vehicle_position_x > 0, lane1))
            if not lane1_distance_front:
                left_front_position_x, left_front_lane, left_front_longitudinalv, left_front_lateralv = ego_vehicle_position_x + 150, -2, 33.3, 0
            else:
                lane1_distance_front_x = [lane1_distance_front[i][0][0] for i in range(len(lane1_distance_front))]
                lane1_distance_front_id = lane1_distance_front[lane1_distance_front_x.index(min(lane1_distance_front_x))][1]
                left_front_position_x, left_front_lane = traci.vehicle.getPosition(lane1_distance_front_id)
                left_front_longitudinalv = traci.vehicle.getSpeed(lane1_distance_front_id)
                left_front_lateralv = traci.vehicle.getLateralSpeed(lane1_distance_front_id)
            lane1_distance_rear = list(filter(lambda x: -150 < x[0][0] - ego_vehicle_position_x < 0, lane1))
            if not lane1_distance_rear:
                left_rear_position_x, left_rear_lane, left_rear_longitudinalv, left_rear_lateralv = ego_vehicle_position_x - 150, -2, 33.3, 0
            else:
                lane1_distance_rear_x = [lane1_distance_rear[i][0][0] for i in range(len(lane1_distance_rear))]
                lane1_distance_rear_id = lane1_distance_rear[lane1_distance_rear_x.index(max(lane1_distance_rear_x))][1]
                left_rear_position_x, left_rear_lane = traci.vehicle.getPosition(lane1_distance_rear_id)
                left_rear_longitudinalv = traci.vehicle.getSpeed(lane1_distance_rear_id)
                left_rear_lateralv = traci.vehicle.getLateralSpeed(lane1_distance_rear_id)
            lane3_distance_front = list(filter(lambda x: 150 > x[0][0] - ego_vehicle_position_x > 0, lane3))
            if not lane3_distance_front:
                right_front_position_x, right_front_lane, right_front_longitudinalv, right_front_lateralv = ego_vehicle_position_x + 150, -10, 27.7, 0
            else:
                lane3_distance_front_x = [lane3_distance_front[i][0][0] for i in range(len(lane3_distance_front))]
                lane3_distance_front_id = lane3_distance_front[lane3_distance_front_x.index(min(lane3_distance_front_x))][1]
                right_front_position_x, right_front_lane = traci.vehicle.getPosition(lane3_distance_front_id)
                right_front_longitudinalv = traci.vehicle.getSpeed(lane3_distance_front_id)
                right_front_lateralv = traci.vehicle.getLateralSpeed(lane3_distance_front_id)
            lane3_distance_rear = list(filter(lambda x: -150 < x[0][0] - ego_vehicle_position_x < 0, lane3))
            if not lane3_distance_rear:
                right_rear_position_x, right_rear_lane, right_rear_longitudinalv, right_rear_lateralv = ego_vehicle_position_x - 150, -10, 27.7, 0
            else:
                lane3_distance_rear_x = [lane3_distance_rear[i][0][0] for i in range(len(lane3_distance_rear))]
                lane3_distance_rear_id = lane3_distance_rear[lane3_distance_rear_x.index(max(lane3_distance_rear_x))][1]
                right_rear_position_x, right_rear_lane = traci.vehicle.getPosition(lane3_distance_rear_id)
                right_rear_longitudinalv = traci.vehicle.getSpeed(lane3_distance_rear_id)
                right_rear_lateralv = traci.vehicle.getLateralSpeed(lane3_distance_rear_id)
        elif -12 <= ego_vehicle_lane < -8:
            lane3_distance_front = list(filter(lambda x: 150 > x[0][0]-ego_vehicle_position_x > 0, lane3))  
            if not lane3_distance_front:
                ego_front_position_x, ego_front_lane, ego_front_longitudinalv, ego_front_lateralv = ego_vehicle_position_x + 150, -10, 27.7, 0
            else:
                lane3_distance_front_x = [lane3_distance_front[i][0][0] for i in range(len(lane3_distance_front))]
                lane3_distance_front_id = lane3_distance_front[lane3_distance_front_x.index(min(lane3_distance_front_x))][1]
                ego_front_position_x, ego_front_lane = traci.vehicle.getPosition(lane3_distance_front_id)
                ego_front_longitudinalv = traci.vehicle.getSpeed(lane3_distance_front_id)
                ego_front_lateralv = traci.vehicle.getLateralSpeed(lane3_distance_front_id)
            lane3_distance_rear = list(filter(lambda x: -150 < x[0][0] - ego_vehicle_position_x < 0, lane3))
            if not lane3_distance_rear:
                ego_rear_position_x, ego_rear_lane, ego_rear_longitudinalv, ego_rear_lateralv = ego_vehicle_position_x - 150, -10, 27.7, 0
            else:
                lane3_distance_rear_x = [lane3_distance_rear[i][0][0] for i in range(len(lane3_distance_rear))]
                lane3_distance_rear_id = lane3_distance_rear[lane3_distance_rear_x.index(max(lane3_distance_rear_x))][1]
                ego_rear_position_x, ego_rear_lane = traci.vehicle.getPosition(lane3_distance_rear_id)
                ego_rear_longitudinalv = traci.vehicle.getSpeed(lane3_distance_rear_id)
                ego_rear_lateralv = traci.vehicle.getLateralSpeed(lane3_distance_rear_id)
            lane2_distance_front = list(filter(lambda x: 150 > x[0][0] - ego_vehicle_position_x > 0, lane2))
            if not lane2_distance_front:
                left_front_position_x, left_front_lane, left_front_longitudinalv, left_front_lateralv = ego_vehicle_position_x + 150, -6, 33.3, 0
            else:
                lane2_distance_front_x = [lane2_distance_front[i][0][0] for i in range(len(lane2_distance_front))]
                lane2_distance_front_id = lane2_distance_front[lane2_distance_front_x.index(min(lane2_distance_front_x))][1]
                left_front_position_x, left_front_lane = traci.vehicle.getPosition(lane2_distance_front_id)
                left_front_longitudinalv = traci.vehicle.getSpeed(lane2_distance_front_id)
                left_front_lateralv = traci.vehicle.getLateralSpeed(lane2_distance_front_id)
            lane2_distance_rear = list(filter(lambda x: -150 < x[0][0] - ego_vehicle_position_x < 0, lane2))
            if not lane2_distance_rear:
                left_rear_position_x, left_rear_lane, left_rear_longitudinalv, left_rear_lateralv = ego_vehicle_position_x - 150, -6, 33.3, 0
            else:
                lane2_distance_rear_x = [lane2_distance_rear[i][0][0] for i in range(len(lane2_distance_rear))]
                lane2_distance_rear_id = lane2_distance_rear[lane2_distance_rear_x.index(max(lane2_distance_rear_x))][1]
                left_rear_position_x, left_rear_lane = traci.vehicle.getPosition(lane2_distance_rear_id)
                left_rear_longitudinalv = traci.vehicle.getSpeed(lane2_distance_rear_id)
                left_rear_lateralv = traci.vehicle.getLateralSpeed(lane2_distance_rear_id)
            lane4_distance_front = list(filter(lambda x: 150 > x[0][0] - ego_vehicle_position_x > 0, lane4))
            if not lane4_distance_front:
                right_front_position_x, right_front_lane, right_front_longitudinalv, right_front_lateralv = ego_vehicle_position_x + 150, -14, 27.7, 0
            else:
                lane4_distance_front_x = [lane4_distance_front[i][0][0] for i in range(len(lane4_distance_front))]
                lane4_distance_front_id = lane4_distance_front[lane4_distance_front_x.index(min(lane4_distance_front_x))][1]
                right_front_position_x, right_front_lane = traci.vehicle.getPosition(lane4_distance_front_id)
                right_front_longitudinalv = traci.vehicle.getSpeed(lane4_distance_front_id)
                right_front_lateralv = traci.vehicle.getLateralSpeed(lane4_distance_front_id)
            lane4_distance_rear = list(filter(lambda x: -150 < x[0][0] - ego_vehicle_position_x < 0, lane4))
            if not lane4_distance_rear:
                right_rear_position_x, right_rear_lane, right_rear_longitudinalv, right_rear_lateralv = ego_vehicle_position_x - 150, -14, 27.7, 0
            else:
                lane4_distance_rear_x = [lane4_distance_rear[i][0][0] for i in range(len(lane4_distance_rear))]
                lane4_distance_rear_id = lane4_distance_rear[lane4_distance_rear_x.index(max(lane4_distance_rear_x))][1]
                right_rear_position_x, right_rear_lane = traci.vehicle.getPosition(lane4_distance_rear_id)
                right_rear_longitudinalv = traci.vehicle.getSpeed(lane4_distance_rear_id)
                right_rear_lateralv = traci.vehicle.getLateralSpeed(lane4_distance_rear_id)

        return ego_front_position_x,ego_front_lane,ego_front_longitudinalv,ego_front_lateralv,ego_rear_position_x,ego_rear_lane,\
                ego_rear_longitudinalv,ego_rear_lateralv,left_front_position_x,left_front_lane,left_front_longitudinalv,left_front_lateralv,\
                left_rear_position_x,left_rear_lane,left_rear_longitudinalv,left_rear_lateralv,right_front_position_x,right_front_lane,\
                right_front_longitudinalv,right_front_lateralv,right_rear_position_x,right_rear_lane,right_rear_longitudinalv,right_rear_lateralv

    def limit_speed(self, ego_vehicle_longitudinalv, ego_vehicle_lane):
        ego_vehicle_longitudinalv = np.array([ego_vehicle_longitudinalv])
        if -16 <= ego_vehicle_lane <= -8:
            ego_vehicle_longitudinalv = np.clip(ego_vehicle_longitudinalv, 0, 27.8)[0]
        else:
            ego_vehicle_longitudinalv = np.clip(ego_vehicle_longitudinalv, 0, 33.3)[0]
        return ego_vehicle_longitudinalv


    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass
        
    def normalization(self,state):
        ego_vehicle_position_x, ego_vehicle_lane, ego_vehicle_longitudinalv, ego_vehicle_lateralv,\
        ego_front_position_x, ego_front_lane, ego_front_longitudinalv, ego_front_lateralv,\
        ego_rear_position_x, ego_rear_lane, ego_rear_longitudinalv, ego_rear_lateralv,\
        left_front_position_x, left_front_lane, left_front_longitudinalv, left_front_lateralv,\
        left_rear_position_x, left_rear_lane, left_rear_longitudinalv, left_rear_lateralv,\
        right_front_position_x, right_front_lane, right_front_longitudinalv, right_front_lateralv,\
        right_rear_position_x, right_rear_lane, right_rear_longitudinalv, right_rear_lateralv = state
        ego_front_position_x = (ego_front_position_x-ego_vehicle_position_x)/150
        ego_front_lane = (ego_front_lane-ego_vehicle_lane)/(-12)
        ego_front_longitudinalv = (ego_front_longitudinalv-ego_vehicle_longitudinalv)/33.3
        ego_front_lateralv = (ego_front_lateralv-ego_vehicle_lateralv)/1.3
        ego_rear_position_x = (ego_rear_position_x-ego_vehicle_position_x)/150
        ego_rear_lane = (ego_rear_lane-ego_vehicle_lane)/(-12)
        ego_rear_longitudinalv = (ego_rear_longitudinalv-ego_vehicle_longitudinalv)/33.3
        ego_rear_lateralv = (ego_rear_lateralv-ego_vehicle_lateralv)/1.3
        left_front_position_x = (left_front_position_x-ego_vehicle_position_x)/150
        left_front_lane = (left_front_lane-ego_vehicle_lane)/(-12)
        left_front_longitudinalv = (left_front_longitudinalv-ego_vehicle_longitudinalv)/33.3
        left_front_lateralv = (left_front_lateralv-ego_vehicle_lateralv)/1.3
        left_rear_position_x = (left_rear_position_x-ego_vehicle_position_x)/150
        left_rear_lane = (left_rear_lane-ego_vehicle_lane)/(-12)
        left_rear_longitudinalv = (left_rear_longitudinalv-ego_vehicle_longitudinalv)/33.3
        left_rear_lateralv = (left_rear_lateralv-ego_vehicle_lateralv)/1.3
        right_front_position_x = (right_front_position_x-ego_vehicle_position_x)/150
        right_front_lane = (right_front_lane-ego_vehicle_lane)/(-12)
        right_front_longitudinalv = (right_front_longitudinalv-ego_vehicle_longitudinalv)/33.3
        right_front_lateralv = (right_front_lateralv-ego_vehicle_lateralv)/1.3
        right_rear_position_x = (right_rear_position_x-ego_vehicle_position_x)/150
        right_rear_lane = (right_rear_lane-ego_vehicle_lane)/(-12)
        right_rear_longitudinalv = (right_rear_longitudinalv-ego_vehicle_longitudinalv)/33.3
        right_rear_lateralv = (right_rear_lateralv-ego_vehicle_lateralv)/1.3
        return np.array([ego_vehicle_position_x/1000, ego_vehicle_lane/(-16), ego_vehicle_longitudinalv/33.3, ego_vehicle_lateralv/1.3,\
        ego_front_position_x, ego_front_lane, ego_front_longitudinalv, ego_front_lateralv,\
        ego_rear_position_x, ego_rear_lane, ego_rear_longitudinalv, ego_rear_lateralv,\
        left_front_position_x, left_front_lane, left_front_longitudinalv, left_front_lateralv,\
        left_rear_position_x, left_rear_lane, left_rear_longitudinalv, left_rear_lateralv,\
        right_front_position_x, right_front_lane, right_front_longitudinalv, right_front_lateralv,\
        right_rear_position_x, right_rear_lane, right_rear_longitudinalv, right_rear_lateralv],dtype = float32)
                

        
        
















