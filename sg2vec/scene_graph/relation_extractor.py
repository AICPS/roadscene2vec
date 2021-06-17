from enum import Enum
import math
import itertools

from .nodes import Node
from .nodes import ObjectNode

#defines all types of actors which can exist
#order of enum values is important as this determines which function is called. DO NOT CHANGE ENUM ORDER
class ActorType(Enum):
    CAR = 0 #26, 142, 137:truck
    MOTO = 1 #80
    BICYCLE = 2 #11
    PED = 3 #90, 91, 98: "player", 78:man, 79:men, 149:woman, 56: guy, 53: girl
    LANE = 4 #124:street, 114:sidewalk
    LIGHT = 5 # 99: "pole", 76: light
    SIGN = 6
    ROAD = 7

class Relations(Enum):
    isIn = 0
    near_coll = 1
    super_near = 2
    very_near = 3
    near = 4
    visible = 5
    inDFrontOf = 6
    inSFrontOf = 7
    atDRearOf = 8
    atSRearOf = 9
    toLeftOf = 10
    toRightOf = 11

#ADD THESE TO VISUALIZER SCRIPT
RELATION_COLORS = ["black", 
                   "red", 
                   "orange", 
                   "yellow",
                   "purple", 
                   "green", 
                   "violet", 
                   "violet",
                   "turquoise", 
                   "turquoise", 
                   "blue", 
                   "blue"]

#This class extracts relations for every pair of entities in a scene
class RelationExtractor:
    def __init__(self, config):
        self.conf = config
        self.CAR_PROXIMITY_THRESH_NEAR_COLL = self.conf.relation_extraction_settings['CAR_PROXIMITY_THRESH_NEAR_COLL']
        # max number of feet between a car and another entity to build proximity relation
        self.CAR_PROXIMITY_THRESH_SUPER_NEAR = self.conf.relation_extraction_settings['CAR_PROXIMITY_THRESH_SUPER_NEAR']
        self.CAR_PROXIMITY_THRESH_VERY_NEAR = self.conf.relation_extraction_settings['CAR_PROXIMITY_THRESH_VERY_NEAR']
        self.CAR_PROXIMITY_THRESH_NEAR = self.conf.relation_extraction_settings['CAR_PROXIMITY_THRESH_NEAR']
        self.CAR_PROXIMITY_THRESH_VISIBLE = self.conf.relation_extraction_settings['CAR_PROXIMITY_THRESH_VISIBLE']
        self.LANE_THRESHOLD = self.conf.relation_extraction_settings['LANE_THRESHOLD'] # feet. if object's center is more than this distance away from ego's center, build left or right lane relation
#         feet. if object's center is within this distance of ego's center, build middle lane relation
        self.CENTER_LANE_THRESHOLD = self.conf.relation_extraction_settings['CENTER_LANE_THRESHOLD']

    def get_actor_type(self, actor):
        if "curr" in actor.attr.keys():
            return ActorType.LANE
        if actor.attr["name"] == "Traffic Light":
            return ActorType.LIGHT
        if actor.attr["name"].split(" ")[0] == "Pedestrian":
            return ActorType.PED
        if actor.attr["name"].split(" ")[0] in self.conf.relation_extraction_settings["CAR_NAMES"]:
            return ActorType.CAR
        if actor.attr["name"].split(" ")[0] in self.conf.relation_extraction_settings["MOTO_NAMES"]:
            return ActorType.MOTO
        if actor.attr["name"].split(" ")[0] in self.conf.relation_extraction_settings["BICYCLE_NAMES"]:
            return ActorType.BICYCLE
        if "Sign" in actor.attr["name"]:
            return ActorType.SIGN

        
        raise NameError("Actor name not found for actor with name: " + actor.attr["name"])

    def get_config(self):
        return self.conf
            
    #takes in two entities and extracts all relations between those two entities. extracted relations are bidirectional    
    def extract_relations(self, actor1, actor2):
        type1 = self.get_actor_type(actor1)
        type2 = self.get_actor_type(actor2)
        
        low_type = min(type1.value, type2.value) #the lower of the two enums.
        high_type = max(type1.value, type2.value)
    
        function_call = "self.extract_relations_"+self.conf.relation_extraction_settings["ACTOR_NAMES"][low_type]+"_"+self.conf.relation_extraction_settings["ACTOR_NAMES"][high_type]+"(actor1, actor2) if type1.value <= type2.value "\
                        "else self.extract_relations_"+self.conf.relation_extraction_settings["ACTOR_NAMES"][low_type]+"_"+self.conf.relation_extraction_settings["ACTOR_NAMES"][high_type]+"(actor2, actor1)"
        return eval(function_call)

    def extract_relative_lanes(self, scene_graph):
        if self.conf.dataset_type == "carla":
            scene_graph.left_lane = Node("lane_left", {"curr":"lane_left"}, ActorType.LANE)
            scene_graph.right_lane = Node("lane_right", {"curr":"lane_right"}, ActorType.LANE)
            scene_graph.middle_lane = Node("lane_middle", {"curr":"lane_middle"}, ActorType.LANE)
        elif self.conf.dataset_type == "image":
            scene_graph.left_lane = ObjectNode('Left Lane', {}, ActorType.LANE)
            scene_graph.right_lane = ObjectNode('Right Lane', {}, ActorType.LANE)
            scene_graph.middle_lane = ObjectNode('Middle Lane', {}, ActorType.LANE)
        scene_graph.add_node(scene_graph.left_lane)
        scene_graph.add_node(scene_graph.right_lane)
        scene_graph.add_node(scene_graph.middle_lane)
        scene_graph.add_relation([scene_graph.left_lane, Relations.isIn, scene_graph.road_node])
        scene_graph.add_relation([scene_graph.right_lane, Relations.isIn, scene_graph.road_node])
        scene_graph.add_relation([scene_graph.middle_lane, Relations.isIn, scene_graph.road_node])
        scene_graph.add_relation([scene_graph.egoNode, Relations.isIn, scene_graph.middle_lane])    
        

    def add_mapping_to_relative_lanes(self, scene_graph, object_node):
        if object_node.label in [ActorType.LANE, ActorType.LIGHT, ActorType.SIGN, ActorType.ROAD]: #don't build lane relations with static objects
            return
        if self.conf.dataset_type == "carla":
            _, ego_y = self.rotate_coords(scene_graph, scene_graph.egoNode.attr['location'][0], scene_graph.egoNode.attr['location'][1]) #NOTE: X corresponds to forward/back displacement and Y corresponds to left/right displacement
            _, new_y = self.rotate_coords(scene_graph, object_node.attr['location'][0], object_node.attr['location'][1])
            y_diff = new_y - ego_y
            if y_diff < -self.LANE_THRESHOLD:
                scene_graph.add_relation([object_node, Relations.isIn, scene_graph.left_lane])
            elif y_diff >  self.LANE_THRESHOLD:
                scene_graph.add_relation([object_node, Relations.isIn, scene_graph.right_lane])
            if abs(y_diff) <= self.CENTER_LANE_THRESHOLD:
                scene_graph.add_relation([object_node, Relations.isIn, scene_graph.middle_lane])
        elif self.conf.dataset_type == "image":
            if object_node.attr['rel_location_x'] < -self.LANE_THRESHOLD:
                scene_graph.add_relation([object_node, Relations.isIn, scene_graph.left_lane])
            elif object_node.attr['rel_location_x'] > self.LANE_THRESHOLD:
                scene_graph.add_relation([object_node, Relations.isIn, scene_graph.right_lane])
            if abs(object_node.attr['rel_location_x']) <= self.CENTER_LANE_THRESHOLD:
                scene_graph.add_relation([object_node, Relations.isIn, scene_graph.middle_lane])

    def extract_semantic_relations(self, scene_graph):
        for node1, node2 in itertools.combinations(scene_graph.g.nodes, 2):
            if node1.name != node2.name: #dont build self-relations
                if node1.type != ActorType.ROAD.value and node2.type != ActorType.ROAD.value:  # dont build relations w/ road
                    scene_graph.add_relations(self.extract_relations(node1, node2))


    #copied from get_node_embeddings(). rotates coordinates to be relative to ego vector.
    def rotate_coords(self, scene_graph, x, y): 
        new_x = (x*scene_graph.ego_cos_term) + (y*scene_graph.ego_sin_term)
        new_y = ((-x)*scene_graph.ego_sin_term) + (y*scene_graph.ego_cos_term)
        return new_x, new_y

#~~~~~~~~~specific relations for each pair of actors possible~~~~~~~~~~~~
#actor 1 corresponds to the first actor in the function name and actor2 the second

    def extract_relations_car_car(self, actor1, actor2):
        relation_list = []
#         if (actor1.name == "car_4" or actor2.name == "car_4"):
#             import pdb; pdb.set_trace()
        # consider the proximity relations with neighboring lanes.

        if self.conf.relation_extraction_settings["extract_all_car_car_relations"] == True:
#             import pdb; pdb.set_trace()
            if self.euclidean_distance(actor1, actor2) <= self.conf.relation_extraction_settings["CAR_PROXIMITY_THRESH_NEAR"]:

                # One of these relations get overwritten in the visualizer for some reason...
                if self.conf.relation_extraction_settings["extract_distance_relations"] == True:
                    relation_list += self.create_proximity_relations(actor1, actor2)
                    relation_list += self.create_proximity_relations(actor2, actor1)
                if self.conf.relation_extraction_settings["extract_directional_relations"] == True:
                    relation_list += self.extract_directional_relation(actor1, actor2)
                    relation_list += self.extract_directional_relation(actor2, actor1)
        else:
            if actor1.name.startswith("ego") or actor2.name.startswith("ego"):
#                 import pdb; pdb.set_trace()
                if self.euclidean_distance(actor1, actor2) <= self.conf.relation_extraction_settings["CAR_PROXIMITY_THRESH_NEAR"]:
                    if self.conf.relation_extraction_settings["extract_distance_relations"] == True:
                        relation_list += self.create_proximity_relations(actor1, actor2)
                        relation_list += self.create_proximity_relations(actor2, actor1)
                    if self.conf.relation_extraction_settings["extract_directional_relations"] == True:
                        relation_list += self.extract_directional_relation(actor1, actor2)
                        relation_list += self.extract_directional_relation(actor2, actor1)
        return relation_list
            
    def extract_relations_car_lane(self, actor1, actor2):
        relation_list = []
        # if(self.in_lane(actor1,actor2)):
        #     relation_list.append([actor1, Relations.isIn, actor2])
            
        return relation_list 
        
    def extract_relations_car_light(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_car_sign(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_car_ped(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_car_bicycle(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_car_moto(self, actor1, actor2):
        relation_list = []
        return relation_list
        
        
    def extract_relations_moto_moto(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_moto_bicycle(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_moto_ped(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_moto_lane(self, actor1, actor2):
        relation_list = []
        # if(self.in_lane(actor1,actor2)):
        #     relation_list.append([actor1, Relations.isIn, actor2])
        #     # relation_list.append([actor2, Relations.isIn, actor1])
        return relation_list 
        
    def extract_relations_moto_light(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_moto_sign(self, actor1, actor2):
        relation_list = []
        return relation_list
        

    def extract_relations_bicycle_bicycle(self, actor1, actor2):
        relation_list = []
        # if(self.euclidean_distance(actor1, actor2) < SELF.CONF["relation_extraction_settings"]["BICYCLE_PROXIMITY_THRESH"]):
        #     relation_list.append([actor1, Relations.near, actor2])
        #     relation_list.append([actor2, Relations.near, actor1])
        #     #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #     #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_bicycle_ped(self, actor1, actor2):
        relation_list = []
        # if(self.euclidean_distance(actor1, actor2) < SELF.CONF["relation_extraction_settings"]["BICYCLE_PROXIMITY_THRESH"]):
        #     relation_list.append([actor1, Relations.near, actor2])
        #     relation_list.append([actor2, Relations.near, actor1])
        #     #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #     #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_bicycle_lane(self, actor1, actor2):
        relation_list = []
        # if(self.in_lane(actor1,actor2)):
        #     relation_list.append([actor1, Relations.isIn, actor2])
        return relation_list 
        
    def extract_relations_bicycle_light(self, actor1, actor2):
        relation_list = []
        #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_bicycle_sign(self, actor1, actor2):
        relation_list = []
        #relation_list.append(self.extract_directional_relation(actor1, actor2))
        #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_ped_ped(self, actor1, actor2):
        relation_list = []
        if(self.euclidean_distance(actor1, actor2) < self.conf.relation_extraction_settings["PED_PROXIMITY_THRESH"]):
            relation_list.append([actor1, Relations.near, actor2])
            relation_list.append([actor2, Relations.near, actor1])
            #relation_list.append(self.extract_directional_relation(actor1, actor2))
            #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
           
    def extract_relations_ped_lane(self, actor1, actor2):
        relation_list = []
        # if(self.in_lane(actor1,actor2)):
        #     relation_list.append([actor1, Relations.isIn, actor2])
        return relation_list 
        
    def extract_relations_ped_light(self, actor1, actor2):
        relation_list = []
        #proximity relation could indicate ped waiting for crosswalk at a light
        # if(self.euclidean_distance(actor1, actor2) < self.conf["relation_extraction_settings"]["PED_PROXIMITY_THRESH"]):
        #     relation_list.append([actor1, Relations.near, actor2])
        #     relation_list.append([actor2, Relations.near, actor1])
            #relation_list.append(self.extract_directional_relation(actor1, actor2))
            #relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_ped_sign(self, actor1, actor2):
        relation_list = []
        # relation_list.append(self.extract_directional_relation(actor1, actor2))
        # relation_list.append(self.extract_directional_relation(actor2, actor1))
        return relation_list
        
    def extract_relations_lane_lane(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_lane_light(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_lane_sign(self, actor1, actor2):
        relation_list = []
        return relation_list

    def extract_relations_light_light(self, actor1, actor2):
        relation_list = []
        return relation_list
        
    def extract_relations_light_sign(self, actor1, actor2):
        relation_list = []
        return relation_list

    def extract_relations_sign_sign(self, actor1, actor2):
        relation_list = []
        return relation_list
    
    
#~~~~~~~~~~~~~~~~~~UTILITY FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~
    #return euclidean distance between actors
    def euclidean_distance(self, actor1, actor2):
        if self.conf.dataset_type == "carla":
            l1 = actor1.attr['location']
            l2 = actor2.attr['location']
            distance = math.sqrt((l1[0] - l2[0])**2 + (l1[1]- l2[1])**2 + (l1[2] - l2[2])**2)
        elif self.conf.dataset_type == "image":
            l1 = (actor1.attr['location_x'], actor1.attr['location_y'])
            l2 = (actor2.attr['location_x'], actor2.attr['location_y'])
            distance = math.sqrt((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)
            # print(actor1, actor2, distance)
        return distance
        
    #check if an actor is in a certain lane
    def in_lane(self, actor1, actor2):
        if 'lane_idx' in actor1.attr.keys():
            # calculate the distance bewteen actor1 and actor2
            # if it is below 3.5 then they have is in relation.
                # if actor1 is ego: if actor2 is not equal to the ego_lane's index then it's invading relation.
            if actor1.attr['lane_idx'] == actor2.attr['lane_idx']:
                return True
            if "invading_lane" in actor1.attr:
                if actor1.attr['invading_lane'] == actor2.attr['lane_idx']:
                    return True
                if "orig_lane_idx" in actor1.attr:
                    if actor1.attr['orig_lane_idx'] == actor2.attr['lane_idx']:
                        return True
        else:
            return False
    
    def create_proximity_relations(self, actor1, actor2):
        if self.euclidean_distance(actor1, actor2) <= self.conf.relation_extraction_settings["CAR_PROXIMITY_THRESH_NEAR_COLL"]:
            return [[actor1, Relations.near_coll, actor2]]
        elif self.euclidean_distance(actor1, actor2) <= self.conf.relation_extraction_settings["CAR_PROXIMITY_THRESH_SUPER_NEAR"]:
            return [[actor1, Relations.super_near, actor2]]
        elif self.euclidean_distance(actor1, actor2) <= self.conf.relation_extraction_settings["CAR_PROXIMITY_THRESH_VERY_NEAR"]:
            return [[actor1, Relations.very_near, actor2]]
        elif self.euclidean_distance(actor1, actor2) <= self.conf.relation_extraction_settings["CAR_PROXIMITY_THRESH_NEAR"]:
            return [[actor1, Relations.near, actor2]]
        elif self.euclidean_distance(actor1, actor2) <= self.conf.relation_extraction_settings["CAR_PROXIMITY_THRESH_VISIBLE"]:
            return [[actor1, Relations.visible, actor2]]
        return []

    def extract_directional_relation(self, actor1, actor2):
        relation_list = []
        if self.conf.dataset_type == "carla":
            # gives directional relations between actors based on their 2D absolute positions.      
            x1, y1 = math.cos(math.radians(actor1.attr['rotation'][0])), math.sin(math.radians(actor1.attr['rotation'][0]))
            x2, y2 = actor2.attr['location'][0] - actor1.attr['location'][0], actor2.attr['location'][1] - actor1.attr['location'][1]
            x2, y2 = x2 / math.sqrt(x2**2+y2**2), y2 / math.sqrt(x2**2+y2**2)
        
        elif self.conf.dataset_type == "image":
            x1 = math.cos(math.radians(0)) 
            y1 = math.sin(math.radians(0))
            x2 = actor2.attr['location_x'] - actor1.attr['location_x']
            y2 = actor2.attr['location_y'] - actor1.attr['location_y']
            x2 /= math.sqrt(x2**2 + y2**2)
            y2 /= math.sqrt(x2**2 + y2**2)

        degree = math.degrees(math.atan2(y1, x1)) - \
                 math.degrees(math.atan2(y2, x2))

        degree %= 360
        

        degree = math.degrees(math.atan2(y1, x1)) - math.degrees(math.atan2(y2, x2)) 
        if degree < 0: 
            degree += 360
            
        if degree <= 45: # actor2 is in front of actor1
            relation_list.append([actor1, Relations.atDRearOf, actor2])
        elif degree >= 45 and degree <= 90:
            relation_list.append([actor1, Relations.atSRearOf, actor2])
        elif degree >= 90 and degree <= 135:
            relation_list.append([actor1, Relations.inSFrontOf, actor2])
        elif degree >= 135 and degree <= 180: # actor2 is behind actor1
            relation_list.append([actor1, Relations.inDFrontOf, actor2])
        elif degree >= 180 and degree <= 225: # actor2 is behind actor1
            relation_list.append([actor1, Relations.inDFrontOf, actor2])
        elif degree >= 225 and degree <= 270:
            relation_list.append([actor1, Relations.inSFrontOf, actor2])
        elif degree >= 270 and degree <= 315:
            relation_list.append([actor1, Relations.atSRearOf, actor2])
        elif degree >= 315 and degree <= 360: 
            relation_list.append([actor1, Relations.atDRearOf, actor2])

        if self.conf.dataset_type == "carla":
            if actor2.attr['lane_idx'] < actor1.attr['lane_idx']: # actor2 to the left of actor1 
                relation_list.append([actor1, Relations.toRightOf, actor2])
            elif actor2.attr['lane_idx'] > actor1.attr['lane_idx']: # actor2 to the right of actor1 
                relation_list.append([actor1, Relations.toLeftOf, actor2])
            
        elif self.conf.dataset_type == "image":  
            if abs(actor2.attr['location_x'] - actor1.attr['location_x']) <= self.CENTER_LANE_THRESHOLD:
                pass
            # actor2 to the left of actor1
            elif actor2.attr['location_x'] < actor1.attr['location_x']:
                relation_list.append([actor2, Relations.toLeftOf, actor1])
            # actor2 to the right of actor1
            elif actor2.attr['location_x'] > actor1.attr['location_x']:
                relation_list.append([actor2, Relations.toRightOf, actor1])
            # disable rear relations help the inference.
            
        
        return relation_list

