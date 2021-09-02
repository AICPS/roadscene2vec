import math
import re
import weakref
from collections import defaultdict

import carla
import numpy as np
import pygame
from carla import ColorConverter as cc


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_attributes(actor, waypoint=None):
    velocity = lambda l: (3.6 * math.sqrt(l.x**2 + l.y**2 + l.z**2))
    # dv = lambda l: (3.6 * math.sqrt((l.x-v.x)**2 + (l.y-v.y)**2 + (l.z-v.z)**2))
    # distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)

    return_dict = defaultdict()
    v_3d = actor.get_velocity()
    t_3d = actor.get_transform()
    l_3d = t_3d.location
    r_3d = t_3d.rotation
    a_3d = actor.get_angular_velocity()

    return_dict['velocity_abs'] = int(velocity(v_3d))
    return_dict['velocity'] = int(v_3d.x), int(v_3d.y), int(v_3d.z)
    return_dict['location'] = int(l_3d.x), int(l_3d.y), int(l_3d.z)
    return_dict['rotation'] =  int(r_3d.yaw), int(r_3d.roll), int(r_3d.pitch)
    return_dict['ang_velocity'] = int(a_3d.x), int(a_3d.y), int(a_3d.z)
    return_dict['name'] = get_actor_display_name(actor)
    if(waypoint):
        return_dict['lane_id'] = waypoint.lane_id
        return_dict['road_id'] = waypoint.road_id
        
    return return_dict
    

def get_vehicle_attributes(vehicle, waypoint=None):
    return_dict = get_actor_attributes(vehicle, waypoint)
    
    light_state = vehicle.get_light_state()
    #light_state variables are booleans
    return_dict['left_blinker_on'] = True if (light_state.LeftBlinker & carla.VehicleLightState.LeftBlinker > 0) else False
    return_dict['right_blinker_on'] = True if (light_state.RightBlinker & carla.VehicleLightState.LeftBlinker > 0) else False
    return_dict['brake_light_on'] = True if (light_state.Brake & carla.VehicleLightState.Brake > 0) else False
    return return_dict


class LaneInvasionDetector(object):
    def __init__(self, parent_actor, storing_path):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionDetector._on_invasion(weak_self, event))

        self.storing_path = storing_path
        self.recording = False

        self.lane_invasion_events = []

    def toggle_recording(self):
        self.recording = not self.recording

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()

    def is_invading_lane(self, frame):
        if self.lane_invasion_events:
            return frame < max(self.lane_invasion_events) + 10
        else:
            return False

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        if self.recording:
            self.lane_invasion_events.append(event.frame)


class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self.collision = False
        self.recording = False
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    def has_collided(self):
        return self.collision
        
    def toggle_recording(self):
        self.recording = not self.recording

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        if self.recording:
            self.collision = True
            impulse = event.normal_impulse
            intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.history.append((event.frame, intensity))
            if len(self.history) > 4000:
                self.history.pop(0)

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()
            
# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, gamma_correction, dimensions, storing_path):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.recording = False
        self.dimensions = dimensions
        self.storing_path = storing_path
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(dimensions[0]))
                bp.set_attribute('image_size_y', str(dimensions[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.dimensions) / 100.0
            lidar_data += (0.5 * self.dimensions[0], 0.5 * self.dimensions[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.dimensions[0], self.dimensions[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype = int)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        
        if self.recording:
            if self.index == 0:
                image.save_to_disk('%s/raw_images/%08d.jpg' % (str(self.storing_path), image.frame))
            else:
                image.save_to_disk('%s/ss_images/%08d.jpg' % (str(self.storing_path), image.frame))

    def destroy(self):
        if self.sensor:
            self.sensor.destroy()