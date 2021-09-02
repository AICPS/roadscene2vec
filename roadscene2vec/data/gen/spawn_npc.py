#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Spawn NPCs into the simulation"""

import glob
import os
import sys
import time

CARLA_API_PATH = r'.\PythonAPI' 
CARLA_ROOT_PATH = r'.\PythonAPI\carla' 
CARLA_DIST_PATH = r'.\PythonAPI\carla\dist'

try:
    sys.path.append(glob.glob('%s/carla-*%d.%d-%s.egg' % (
        CARLA_DIST_PATH,
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])

except IndexError:
    print('Import failed! path: %s/carla-*%d.%d-%s.egg' % (
        CARLA_DIST_PATH,
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))

sys.path.append(CARLA_API_PATH)
sys.path.append(CARLA_DIST_PATH)
sys.path.append(CARLA_ROOT_PATH)

import carla

import argparse
import logging

from lane_change_recorder import *

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class CarlaWorld():

    def __init__(self, args):
        self.args = args
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(20.0)

        # Town 5: Squared-grid town with cross junctions and a bridge. 
        # It has multiple lanes per direction. Useful to perform lane changes.
        self.world = self.client.load_world('Town04')
        CarlaDataProvider.set_world(self.world)
        
        # set number of vehicles randomly
        self.number_of_vehicles = random.randrange(max(2, self.args.number_of_vehicles / 2), self.args.number_of_vehicles)
        # set number of pedestrians randomly 
        self.number_of_walkers = random.randrange(1, self.args.number_of_walkers)

        self.lanechangerecorder = None

    def run(self):

        traffic_manager = self.client.get_trafficmanager(self.args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        self.world = self.client.get_world()

        self.synchronous_master = False

        if self.args.sync:
            settings = self.world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                self.synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(settings)
            else:
                self.synchronous_master = False

        blueprints = self.world.get_blueprint_library().filter(self.args.filterv)
        blueprintsWalkers = self.world.get_blueprint_library().filter(self.args.filterw)

        if self.args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if self.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif self.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, self.number_of_vehicles, number_of_spawn_points)
            self.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= self.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in self.client.apply_batch_sync(batch, self.synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = self.world.get_actors(self.all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not self.args.sync or not self.synchronous_master:
            self.world.wait_for_tick()
        else:
            self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(self.vehicles_list), len(self.walkers_list)))

        # change global vehicle behavior
        for v in self.vehicles_list:
            # disable auto lane change
            traffic_manager.auto_lane_change(self.world.get_actor(v), False)
            traffic_manager.vehicle_percentage_speed_difference(self.world.get_actor(v), random.uniform(-20, 30))
            traffic_manager.distance_to_leading_vehicle(self.world.get_actor(v), random.uniform(0.5, 50))

        # if you want to trigger the recorder, run this file in synchronous mode
        self.lanechangerecorder = LaneChangeRecorder(traffic_manager, self.world, self.client)
        self.lanechangerecorder.set_vehicles_list(self.vehicles_list)

        while True:

            timestamp = None
            reset = False

            if self.args.sync and self.synchronous_master:
                self.world.tick()
            else:
                self.world.wait_for_tick()

            if self.world: 
                snapshot = self.world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            try:
                reset = self.lanechangerecorder.tick(timestamp.frame)
            except Exception as e:
                raise e

            if reset:
                break
    
    def clean(self):
        if self.args.sync and self.synchronous_master:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        
        if self.lanechangerecorder:
            self.lanechangerecorder.destroy_sensors()
        self.lanechangerecorder = None
        
        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('\ndestroying %d walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        
        time.sleep(0.5)

    def reload(self):
        self.world = self.client.reload_world()
        # set number of vehicles randomly
        self.number_of_vehicles = random.randrange(max(2, self.args.number_of_vehicles / 2), self.args.number_of_vehicles)
        # set number of pedestrians randomly 
        self.number_of_walkers = random.randrange(1, self.args.number_of_walkers)

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='max number of vehicles (default: 10)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=50,
        type=int,
        help='max number of walkers (default: 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '-tm_p', '--tm_port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    
    carla_world = CarlaWorld(args)

    try:
        while True:
            try:
                carla_world.run()
            except KeyboardInterrupt as ki:
                raise ki
            except Exception as e:
                print(e)
                pass
            carla_world.clean()
            carla_world.reload()
    finally:
        carla_world.clean()

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
