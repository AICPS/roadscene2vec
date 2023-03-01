import math
import itertools
from roadscene2vec.scene_graph.nodes import Node


class RelationExtractor:
    '''extracts relations for every pair of entities in a scene'''
    def __init__(self, config):
        self.conf = config
        self.actors = config.relation_extraction_settings["ACTOR_NAMES"]
        self.rels = config.relation_extraction_settings["RELATION_NAMES"]
        self.wanted_directional_relation_dict = {(i[0],i[1]):i[2] for i in config.relation_extraction_settings["directional_relation_list"]}
        self.wanted_proximity_relation_dict = {(i[0],i[1]):i[2] for i in config.relation_extraction_settings["proximity_relation_list"]}
        self.proximity_rels = self.conf.relation_extraction_settings["PROXIMITY_THRESHOLDS"]
        self.directional_rels = config.relation_extraction_settings["DIRECTIONAL_THRESHOLDS"]
        self.relational_colors = {i[0]:i[1] for i in config.relation_extraction_settings["RELATION_COLORS"]}
        self.LANE_THRESHOLD = self.conf.relation_extraction_settings['LANE_THRESHOLD'] # feet. if object's center is more than this distance away from ego's center, build left or right lane relation
        #self.CENTER_LANE_THRESHOLD = self.conf.relation_extraction_settings['CENTER_LANE_THRESHOLD']# feet. if object's center is within this distance of ego's center, build middle lane relation

    def get_actor_type(self, actor):
        for actor_ in range(len(self.actors)):
            if actor.label == self.actors[actor_]:
                return self.actors[actor_], actor_ #return the actor type along with its index in the ACTOR_NAMES list
            elif actor.label.lower() == self.actors[actor_]:
                return self.actors[actor_], actor_
            elif f"{self.actors[actor_].upper()}_NAMES" in self.conf.relation_extraction_settings:
              for actor_names in self.conf.relation_extraction_settings[f"{self.actors[actor_].upper()}_NAMES"]: #go through different names of actor type (ie Tesla for type car)
                  if actor_names in actor.label:
                      return self.actors[actor_], actor_
                  elif actor_names in actor.label.lower():
                      return self.actors[actor_], actor_
        raise NameError("Actor name not found for actor with name: " + actor.attr["name"])

    def get_config(self):
        return self.conf
            

    def extract_relations(self, actor1, actor2):
        '''takes in two entities and extracts all relations between those two entities. extracted relations are bidirectional'''
        type1 ,_ = self.get_actor_type(actor1)
        type2 ,_= self.get_actor_type(actor2)
        relations_list = []
        if (type1,type2) in self.wanted_proximity_relation_dict.keys():
            relations_list += self.extract_distance_relations_actor1_actor2(actor1, actor2, type1, type2) #always pass in order that they are defined in the list
        if (type1,type2) in self.wanted_directional_relation_dict.keys():
            relations_list += self.extract_directional_relation_actor1_actor2(actor1, actor2, type1, type2) #always pass in order that they are defined in the list
        return relations_list
        

    def extract_relative_lanes(self, scene_graph): #keep as you will always need to add lanes
        if self.conf.dataset_type == "carla":
            scene_graph.left_lane = Node("lane_left", {"curr":"lane_left"}, "lane",  self.actors.index("lane")) #change actor.lane to just lane 
            scene_graph.right_lane = Node("lane_right", {"curr":"lane_right"}, "lane",  self.actors.index("lane"))
            scene_graph.middle_lane = Node("lane_middle", {"curr":"lane_middle"}, "lane",  self.actors.index("lane"))
        elif self.conf.dataset_type == "image":
            scene_graph.left_lane = Node('Left Lane', {}, "lane",  self.actors.index("lane"))
            scene_graph.right_lane = Node('Right Lane', {}, "lane",  self.actors.index("lane"))
            scene_graph.middle_lane = Node('Middle Lane', {}, "lane",  self.actors.index("lane"))
        scene_graph.add_node(scene_graph.left_lane)
        scene_graph.add_node(scene_graph.right_lane)
        scene_graph.add_node(scene_graph.middle_lane)

        scene_graph.add_relation([scene_graph.left_lane, "isIn", scene_graph.road_node]) #if we assume lanes and roads must be in graph, then just check to see if isIn in the wanted relations?
        scene_graph.add_relation([scene_graph.right_lane, "isIn", scene_graph.road_node])
        scene_graph.add_relation([scene_graph.middle_lane, "isIn", scene_graph.road_node])
        scene_graph.add_relation([scene_graph.egoNode, "isIn", scene_graph.middle_lane])    


    def add_mapping_to_relative_lanes(self, scene_graph, object_node): #leave this in if we can assume that there will always be lanes
        if self.conf.dataset_type == "carla":
            _, ego_y = self.rotate_coords(scene_graph, scene_graph.egoNode.attr['location'][0], scene_graph.egoNode.attr['location'][1]) #NOTE: X corresponds to forward/back displacement and Y corresponds to left/right displacement
            _, new_y = self.rotate_coords(scene_graph, object_node.attr['location'][0], object_node.attr['location'][1])
            y_diff = new_y - ego_y
            if y_diff < -self.LANE_THRESHOLD:
                scene_graph.add_relation([object_node, "isIn", scene_graph.left_lane])
            elif y_diff >  self.LANE_THRESHOLD:
                scene_graph.add_relation([object_node, "isIn", scene_graph.right_lane])
            elif y_diff <= self.LANE_THRESHOLD and y_diff >= -self.LANE_THRESHOLD: #check
                scene_graph.add_relation([object_node, "isIn", scene_graph.middle_lane])
#            elif abs(y_diff) <= self.CENTER_LANE_THRESHOLD:
#                scene_graph.add_relation([object_node, "isIn", scene_graph.middle_lane])
        elif self.conf.dataset_type == "image": 
            if object_node.attr['rel_location_x'] < -self.LANE_THRESHOLD:
                scene_graph.add_relation([object_node, "isIn", scene_graph.left_lane]) 
            elif object_node.attr['rel_location_x'] > self.LANE_THRESHOLD:
                scene_graph.add_relation([object_node, "isIn", scene_graph.right_lane])
#            elif abs(object_node.attr['rel_location_x']) <= self.CENTER_LANE_THRESHOLD:
#                scene_graph.add_relation([object_node, "isIn", scene_graph.middle_lane])
            elif object_node.attr['rel_location_x'] <= self.LANE_THRESHOLD and object_node.attr['rel_location_x'] >= -self.LANE_THRESHOLD:
                scene_graph.add_relation([object_node, "isIn", scene_graph.middle_lane])

    def extract_semantic_relations(self, scene_graph):
        for node1, node2 in itertools.combinations(scene_graph.g.nodes, 2):
            if node1.name != node2.name and (node1.name != "Root Road" and node2.name != "Root Road"): #dont build self-relations
                scene_graph.add_relations(self.extract_relations(node1, node2))
                

    def rotate_coords(self, scene_graph, x, y): 
        '''copied from get_node_embeddings(). rotates coordinates to be relative to ego vector.'''
        new_x = (x*scene_graph.ego_cos_term) + (y*scene_graph.ego_sin_term)
        new_y = ((-x)*scene_graph.ego_sin_term) + (y*scene_graph.ego_cos_term)
        return new_x, new_y

#~~~~~~~~~specific relations for each pair of actors possible~~~~~~~~~~~~
#actor 1 corresponds to the first actor in the function name and actor2 the second

    def extract_distance_relations_actor1_actor2(self, actor1, actor2, type1, type2):
        relation_list = []
        if self.euclidean_distance(actor1, actor2) <= self.wanted_proximity_relation_dict[(type1,type2)]:
            relation_list += self.create_proximity_relations(actor1, actor2)
            relation_list += self.create_proximity_relations(actor2, actor1)
            return relation_list
        return relation_list


    def extract_directional_relation_actor1_actor2(self, actor1, actor2, type1, type2):
        relation_list = []
        if self.euclidean_distance(actor1, actor2) <= self.wanted_directional_relation_dict[(type1,type2)]:
            # One of these relations get overwritten in the visualizer for some reason...
            relation_list += self.extract_directional_relation(actor1, actor2)
            relation_list += self.extract_directional_relation(actor2, actor1)
            return relation_list
        return relation_list
    
#~~~~~~~~~~~~~~~~~~UTILITY FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~
    
    def euclidean_distance(self, actor1, actor2):
        '''return euclidean distance between actors'''
        if self.conf.dataset_type == "carla":
            l1 = actor1.attr['location']
            l2 = actor2.attr['location']
            distance = math.sqrt((l1[0] - l2[0])**2 + (l1[1]- l2[1])**2 + (l1[2] - l2[2])**2)
        elif self.conf.dataset_type == "image":
            l1 = (actor1.attr['location_x'], actor1.attr['location_y'])
            l2 = (actor2.attr['location_x'], actor2.attr['location_y'])
            distance = math.sqrt((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)
        return distance
        

    def in_lane(self, actor1, actor2):
        '''check if an actor is in a certain lane'''
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
        for relation in self.proximity_rels:
            if self.euclidean_distance(actor1, actor2) <= relation[1]:
                return [[actor1,relation[0], actor2]]
        return []

    def extract_directional_relation(self, actor1, actor2):
        relation_list = []
        if self.conf.dataset_type == "carla":
            # gives directional relations between actors based on their 2D absolute positions.      
            x1, y1 = math.cos(math.radians(actor1.attr['rotation'][0])), math.sin(math.radians(actor1.attr['rotation'][0]))
            x2, y2 = actor2.attr['location'][0] - actor1.attr['location'][0], actor2.attr['location'][1] - actor1.attr['location'][1]
            x2, y2 = x2 / math.sqrt(x2**2+y2**2), y2 / math.sqrt(x2**2+y2**2)
            
            degree =  math.degrees(math.atan2(y2, x2)) - \
                 math.degrees(math.atan2(y1, x1))
        
        elif self.conf.dataset_type == "image":
            x1 = math.cos(math.radians(0)) 
            y1 = math.sin(math.radians(0))
            x2 = actor2.attr['location_x'] - actor1.attr['location_x']
            y2 = actor2.attr['location_y'] - actor1.attr['location_y']
            x2 /= math.sqrt(x2**2 + y2**2)
            y2 /= math.sqrt(x2**2 + y2**2)
      
            degree = math.degrees(math.atan2(y1, x1)) - \
                 math.degrees(math.atan2(y2, x2))
        
        if degree < 0: 
            degree += 360
        degree %= 360
             
        for direction_rel in self.directional_rels:
            list_of_ranges = direction_rel[1]
            for ranges in list_of_ranges:
                if degree >= ranges[0] and degree <= ranges[1]:
                    relation_list.append([actor2, direction_rel[0], actor1])           
    
        if self.conf.dataset_type == "carla":
            if actor2.attr['lane_idx'] < actor1.attr['lane_idx']: # actor2 to the left of actor1 
                relation_list.append([actor2, "toLeftOf", actor1])
            elif actor2.attr['lane_idx'] > actor1.attr['lane_idx']: # actor2 to the right of actor1 
                relation_list.append([actor2, "toRightOf", actor1])
               
        elif self.conf.dataset_type == "image":  
            if (actor2.attr['location_x'] - actor1.attr['location_x']) <= self.LANE_THRESHOLD and (actor2.attr['location_x'] - actor1.attr['location_x']) >= -self.LANE_THRESHOLD: #if in the same lane, don't want left or right relations to be built
                pass
            # actor2 to the left of actor1
            elif actor2.attr['location_x'] < actor1.attr['location_x']:
                relation_list.append([actor2, "toLeftOf", actor1])
            # actor2 to the right of actor1
            elif actor2.attr['location_x'] > actor1.attr['location_x']:
                relation_list.append([actor2, "toRightOf", actor1])
            # disable rear relations help the inference.
             
        return relation_list


