import carla
import json
import math
import py_trees
import random
import sys
from collections import defaultdict
from pathlib import Path

import imageio

import sensors
from sensors import get_actor_attributes, get_vehicle_attributes

SRUNNER_PATH = r'./scenario_runner'
sys.path.append(SRUNNER_PATH)
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import LaneChange


class LaneChangeRecorder:

    def __init__(self, traffic_manager, carla_world, client):
        # It has Idle, 
        self.state = "Idle"
        self.tick_count = 0
        self.carla_world = carla_world
        self.map = self.carla_world.get_map()
        self.vehicles_list = []
        self.traffic_manager = traffic_manager
        self.sensors_dict = {}
        self.root_path = Path("./_out")

        self.root_path.mkdir(exist_ok=True)
        self.new_path = None
        
        self.num_of_existing_datapoints = len(list(self.root_path.glob('*')))
        self.dir_index = 0

        # This indicates the state of ego's lane change behavior
        self.lane_changing= False

        self.client = client

        self.weather_presets = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.CloudyNoon, 
                carla.WeatherParameters.WetNoon, carla.WeatherParameters.WetCloudyNoon, 
                carla.WeatherParameters.SoftRainNoon, carla.WeatherParameters.MidRainyNoon, 
                carla.WeatherParameters.HardRainNoon, carla.WeatherParameters.ClearSunset, 
                carla.WeatherParameters.CloudySunset, carla.WeatherParameters.WetSunset, 
                carla.WeatherParameters.WetCloudySunset, carla.WeatherParameters.SoftRainSunset, 
                carla.WeatherParameters.MidRainSunset, carla.WeatherParameters.HardRainSunset]

        self.recording_count = 0

    def set_vehicles_list(self, vehicles_list):
        self.vehicles_list = vehicles_list

    def attach_sensors(self, root_path):
        """
        Spawn and attach sensors to ego vehicles
        """
        cam_index = 0
        cam_pos_index = 1
        dimensions = [1280, 720]
        gamma = 2.2

        self.sensors_dict["camera_manager"] = sensors.CameraManager(self.ego, gamma, dimensions, root_path)
        self.sensors_dict["camera_manager"].transform_index = cam_pos_index
        self.sensors_dict["camera_manager"].set_sensor(cam_index, notify=False)
        self.sensors_dict["lane_invasion"] = sensors.LaneInvasionDetector(self.ego, root_path)
        self.sensors_dict["collision"] = sensors.CollisionSensor(self.ego)
        # self.sensors_dict["camera_manager_ss"] = sensors.CameraManager(self.ego, gamma, dimensions, root_path)
        # self.sensors_dict["camera_manager_ss"].transform_index = cam_pos_index
        # self.sensors_dict["camera_manager_ss"].set_sensor(cam_index+5, notify=False)

    def destroy_sensors(self):
        for _, sensor in self.sensors_dict.items():
            sensor.destroy()
        self.sensors_dict = {}
        
    def toggle_recording(self):
        for _, sensor in self.sensors_dict.items():
            sensor.toggle_recording()
    
    def convert_gif(self, path):
        path = Path(path).resolve()
        folder_path = path / 'raw_images'
        img_path = folder_path.glob('**/*.jpg')
        images = []
        for filename in img_path:
            images.append(imageio.imread(str(filename)))
        imageio.mimsave(path / 'lane_change.gif', images, format='GIF')

    def tick(self, frame_num):
        self.tick_count += 1
        
        if self.tick_count == 100:
            # set random weather
            self.carla_world.set_weather(random.choice(self.weather_presets))

            # choose random vehicle and prepare for recording
            print("Picking a vehicle...")
            self.ego = self.carla_world.get_actor(random.choice(self.vehicles_list))
            spetator_transform = self.ego.get_transform()
            spetator_transform.location.z += 3 
            if abs(spetator_transform.rotation.yaw) > abs(spetator_transform.rotation.pitch):
                if spetator_transform.rotation.yaw > 0:
                    spetator_transform.location.y -= 3 
                else:
                    spetator_transform.location.y += 3
            else: 
                if spetator_transform.rotation.pitch > 0:
                    spetator_transform.location.x -= 3 
                else:
                    spetator_transform.location.x += 3

            self.carla_world.get_spectator().set_transform(spetator_transform)

            print("Attempting lane change...")
            self.lane_change_direction = None
            
            # check available lane changes
            waypoint = self.map.get_waypoint(self.ego.get_location())
            velocity = self.ego.get_velocity()

            if ( waypoint.lane_change == carla.LaneChange.NONE or 
                (abs(velocity.x) <= 1.0 and abs(velocity.y) <= 1.0) ):
                print("Lane Change not available.")
                self.tick_count = 0
                return
            elif (waypoint.lane_change == carla.LaneChange.Both):
                print("Both")
                self.lane_change_direction = random.choice(['left', 'right'])
            elif (waypoint.lane_change == carla.LaneChange.Left):
                print("Left")
                self.lane_change_direction = 'left'
            else:
                print("Right")
                self.lane_change_direction = 'right'

            print("Start Lane Changing and Recording...")
            self.lane_changing= True 

            self.new_path = "%s/%s" % (str(self.root_path), self.num_of_existing_datapoints + self.dir_index)
            self.extractor = DataExtractor(self.ego, self.new_path)
            self.attach_sensors(self.new_path)
            self.dir_index += 1
            self.toggle_recording()

            ## setting 
            abs_velocity = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            self.lane_change_controller = LaneChange(self.ego, speed=abs_velocity, direction=self.lane_change_direction, distance_other_lane=50)
            self.lane_change_controller.initialise()

            self.client.apply_batch_sync([carla.command.SetAutopilot(self.ego, False)], True)
              
        if self.lane_changing:
            lane_invasion = self.sensors_dict["lane_invasion"].is_invading_lane(frame_num)
            self.extractor.extract_frame(self.carla_world, self.map, frame_num, \
                                         lane_invasion=lane_invasion,\
                                         lane_change_direction=self.lane_change_direction)
            success = self.lane_change_controller.update()
            if (success == py_trees.common.Status.SUCCESS 
                or self.sensors_dict['collision'].has_collided()
                or self.tick_count > 200):
                #write to metadata file
                if self.sensors_dict['collision'].has_collided():
                    print("Collision")
                self.toggle_recording()
                self.destroy_sensors()
                with open((Path(self.new_path) / 'metadata.txt').resolve(),'w') as file:
                    weather=self.carla_world.get_weather()
                    
                    metadata_dict={"wetness":weather.wetness,"wind_intensity":weather.wind_intensity,"precipitation_deposits":weather.precipitation_deposits,
                    "precipitation": weather.precipitation,"cloudiness": weather.cloudiness,"fog_density": weather.fog_density,"fog_distance": weather.fog_distance,
                    "sun_altitude_angle": weather.sun_altitude_angle,"sun_azimuth_angle": weather.sun_azimuth_angle, "lane_change_direction": self.lane_change_direction}
                    
                    file.write(json.dumps(metadata_dict))

                self.extractor.export_data()
                # create gifs
                # self.convert_gif(self.new_path)
                self.lane_changing = False
                print('set set_autopilot back to true')
                self.client.apply_batch_sync([carla.command.SetAutopilot(self.ego, True)], True)
                print("Cleaning up sensors...")
                self.tick_count = 0
                self.recording_count += 1
        
        return self.recording_count >= 10

class DataExtractor(object):

    def __init__(self, ego, store_path):
        
        self.output_root_dir = Path(store_path).resolve()
        self.output_dir = (Path(store_path) / 'scene_raw').resolve()

        self.output_root_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        self.framedict=defaultdict()
        self.ego = ego
        self.orig_ego_lane_idx = None

    def extract_frame(self, world, map1, frame, lane_invasion=False, lane_change_direction="left"):
        t = self.ego.get_transform()
        ego_location = self.ego.get_location()
        distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)

        vehicles = world.get_actors().filter('vehicle.*')
        pedestrians = world.get_actors().filter('walker.*')
        trafficlights = world.get_actors().filter('traffic.traffic_light')
        signs = world.get_actors().filter('traffic.traffic_sign')
        
        egodict = defaultdict()
        actordict = defaultdict()
        peddict = defaultdict()
        lightdict = defaultdict()
        signdict = defaultdict()
        lanedict = defaultdict()

        waypoint = map1.get_waypoint(ego_location, project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))

        ego_lane = waypoint
                
        def build_dict_lane_single(lane_waypoint):
            return {
                'location_x': lane_waypoint.transform.location.x,
                'location_y': lane_waypoint.transform.location.y,
                'location_z': lane_waypoint.transform.location.z,
                'lane_id': lane_waypoint.lane_id,
                'road_id': lane_waypoint.road_id, 
                'lane_type': lane_waypoint.lane_type.name, 
                'lane_width': lane_waypoint.lane_width, 
                'right_lane_marking_type': lane_waypoint.right_lane_marking.type.name, 
                'left_lane_marking_type': lane_waypoint.left_lane_marking.type.name,
                'lane_change': lane_waypoint.lane_change.name,
                'is_junction': lane_waypoint.is_junction,
                ### adding transform if possible.
            }

        def build_dict_lane(lane_waypoint, distance=100):
            ## called by each neighboring lane.
            lane_dict = {}
            lane_dict["curr"]  = [build_dict_lane_single(lane_waypoint)]
            lane_dict["next"] = [build_dict_lane_single(next_waypoint) for next_waypoint in lane_waypoint.next(distance)]
            lane_dict["prev"] = [build_dict_lane_single(next_waypoint) for next_waypoint in lane_waypoint.previous(distance)]
            return lane_dict

        def build_lanes(src_lane, direction="left"):
            lanes = []
            cur_lane = src_lane # starting the src_lane (ego)
            while True:
                lane = cur_lane.get_left_lane() if direction == "left" else cur_lane.get_right_lane()
                if lane is None:
                    break 
                if lane.lane_type in [carla.LaneType.Shoulder, carla.LaneType.Sidewalk]:
                    break
                if cur_lane.lane_id * lane.lane_id < 0: ## special handling.
                    break
                lanes.append(build_dict_lane(lane))
                cur_lane = lane
            return lanes
        
        def get_actor_lane_idx(lanes, lane_id, road_id):
            for idx, lane in enumerate(lanes):
                for key, lane_list in lane.items():
                    for lane_dict in lane_list:
                        if lane_dict['lane_id'] == lane_id and lane_dict['road_id'] == road_id:
                            return idx, key
            return None, None

        # 1. build the road topology based where the ego is at. 
        # 2. build the lane idx by our own. not using the opendriving idx.
        #    also we store waypoints of next(100) and previous(100) for each lane. 

        # lane 0: [current, next1, next2, previous 1, previous 2] next(50), 
        # lane 1: [current, next1, next2, previous 1, previous 2]
        # lane 2: ego lane [current, next1, next2, previous 1, previous 2 

        ## build the new lane dictionary and systems. 
        left_lanes = build_lanes(waypoint)[::-1]
        right_lanes = build_lanes(waypoint, direction="right")
        lanes = left_lanes + [build_dict_lane(ego_lane)] + right_lanes
        lanedict['lanes'] = lanes
        lanedict['ego_lane_idx'] = len(left_lanes)
        # import pdb; pdb.set_trace()

        egodict = get_vehicle_attributes(self.ego, waypoint)
        egodict['lane_idx'] = lanedict['ego_lane_idx'] 
        if self.orig_ego_lane_idx == None:
            self.orig_ego_lane_idx = lanedict['ego_lane_idx'] 
        egodict['orig_lane_idx'] =  self.orig_ego_lane_idx 
        if lane_invasion:
            if lane_change_direction == "left":
                lane_id = self.orig_ego_lane_idx - 1
            else:
                lane_id = self.orig_ego_lane_idx + 1
            egodict["invading_lane"] = lane_id

        # export data from surrounding vehicles
        if len(vehicles) > 1:
            for vehicle in vehicles:
                # TODO: change the 100m condition to field of view. 
                if vehicle.id != self.ego.id and distance(vehicle.get_location()) < 100:
                    vehicle_wp = map1.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
                    vehicle_dict = get_vehicle_attributes(vehicle, vehicle_wp)
                    vehicle_dict["lane_idx"], vehicle_dict["relative_position"] = get_actor_lane_idx(lanes, vehicle_dict['lane_id'], vehicle_dict['road_id'])# the found lane_idx
                    if vehicle_dict["lane_idx"] is not None:
                        actordict[vehicle.id] = vehicle_dict
    
        for p in pedestrians:
            if p.get_location().distance(self.ego.get_location()) < 100:
                ped_wp = map1.get_waypoint(p.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
                ped_dict = get_actor_attributes(p, ped_wp)
                ped_dict["lane_idx"], ped_dict["relative_position"] = get_actor_lane_idx(lanes, ped_dict['lane_id'], ped_dict['road_id'])# the found lane_idx
                if ped_dict["lane_idx"] is not None:
                    peddict[p.id] = ped_dict

        for t_light in trafficlights:
            if t_light.get_location().distance(self.ego.get_location()) < 100:
                lightdict[t_light.id]=get_actor_attributes(t_light)

        for s in signs:
            if s.get_location().distance(self.ego.get_location()) < 100:
                signdict[s.id]=get_actor_attributes(s)

        self.framedict[frame]={"ego": egodict,"actors": actordict,"pedestrians": peddict,"trafficlights": lightdict,"signs": signdict,"lane": lanedict}
        
    def export_data(self):
        with open(self.output_dir / (str(list(self.framedict.keys())[0]) + '-' + str(list(self.framedict.keys())[len(self.framedict)-1])+'.json'), 'w') as file:
            file.write(json.dumps(self.framedict))
        self.framedict.clear()
