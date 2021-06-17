import itertools
import numpy as np
import networkx as nx
import os, cv2, sys, math
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import to_agraph
from .relation_extractor import ActorType, Relations, RELATION_COLORS #don't need RelationExtractor, but still need these
sys.path.append(os.path.dirname(sys.path[0]))


class ObjectNode:
    def __init__(self, name, attr, label):
        self.name = name  # Car-1, Car-2.
        self.attr = attr  # bounding box info
        self.label = label  # ActorType

    def __repr__(self):
        return '%s' % (self.name)


'''Create scenegraph using raw image frame tensor'''
class RealSceneGraph:
    ''' 
        scene graph the real images 
        arguments: 
            image_path : path to the image for which the scene graph is generated

    '''

    def __init__(self, relation_extractor, image_tensor, bounding_boxes, bev, coco_class_names=None, platform='image'):
        self.relation_extractor = relation_extractor
        
        #configure image settings
        self.IMAGE_H = self.relation_extractor.get_config().image_setttings['IMAGE_H']
        self.IMAGE_W = self.relation_extractor.get_config().image_setttings['IMAGE_W']
        self.CROPPED_H = self.relation_extractor.get_config().image_setttings['CROPPED_H']
        self.BIRDS_EYE_IMAGE_H = self.relation_extractor.get_config().image_setttings['BIRDS_EYE_IMAGE_H']
        self.BIRDS_EYE_IMAGE_W = self.relation_extractor.get_config().image_setttings['BIRDS_EYE_IMAGE_W']
        self.Y_SCALE = self.relation_extractor.get_config().image_setttings['Y_SCALE'] # 22 pixels = length of lane line (10 feet)
        self.X_SCALE = self.relation_extractor.get_config().image_setttings['X_SCALE'] # 26 pixels = width of lane (12 feet)
#         self.H_OFFSET = self.IMAGE_H - self.CROPPED_H  # offset from top of image to start of ROI
#         
        #configure relation extraction settings
        self.CAR_PROXIMITY_THRESH_NEAR_COLL = self.relation_extractor.get_config().relation_extraction_settings['CAR_PROXIMITY_THRESH_NEAR_COLL']
        # max number of feet between a car and another entity to build proximity relation
        self.CAR_PROXIMITY_THRESH_SUPER_NEAR = self.relation_extractor.get_config().relation_extraction_settings['CAR_PROXIMITY_THRESH_SUPER_NEAR']
        self.CAR_PROXIMITY_THRESH_VERY_NEAR = self.relation_extractor.get_config().relation_extraction_settings['CAR_PROXIMITY_THRESH_VERY_NEAR']
        self.CAR_PROXIMITY_THRESH_NEAR = self.relation_extractor.get_config().relation_extraction_settings['CAR_PROXIMITY_THRESH_NEAR']
        self.CAR_PROXIMITY_THRESH_VISIBLE = self.relation_extractor.get_config().relation_extraction_settings['CAR_PROXIMITY_THRESH_VISIBLE']
        self.LANE_THRESHOLD = self.relation_extractor.get_config().relation_extraction_settings['LANE_THRESHOLD'] # feet. if object's center is more than this distance away from ego's center, build left or right lane relation
        # feet. if object's center is within this distance of ego's center, build middle lane relation
        self.CENTER_LANE_THRESHOLD = self.relation_extractor.get_config().relation_extraction_settings['CENTER_LANE_THRESHOLD']


        #should we use config for this below?
        self.actors = {'car':           ActorType.CAR,
                       'truck':         ActorType.CAR,
                       'bus':           ActorType.CAR,
                       }
                    #    'person':        ActorType.PED,
                    #    'bicycle':       ActorType.BICYCLE,
                    #    'motorcycle':    ActorType.MOTO, 
                    #    'traffic light': ActorType.LIGHT,
                    #    'stop sign':     ActorType.SIGN,
                    #     }

        self.bev = bev
        
        self.g = nx.MultiDiGraph()  # initialize scenegraph as networkx graph
        self.image_tensor = image_tensor
        # road and lane settings.
        # we need to define the type of node.
        self.road_node = ObjectNode('Root Road', {}, ActorType.ROAD)
        self.add_node(self.road_node)   # adding the road as the root node

        # specify which type of data to load into model (options: image or honda)
        self.platfrom = platform

        # set ego location to middle-bottom of image.
        self.ego_location = (self.bev.params['width'] * self.X_SCALE, 
                            (self.bev.params['height'] + self.bev.params['cropped_height']) * self.Y_SCALE)

        self.ego_node = ObjectNode('Ego Car', {
                                   'location_x': self.ego_location[0], 
                                   'location_y': self.ego_location[1]}, 
                                   ActorType.CAR)

        self.add_node(self.ego_node)
        self.extract_relative_lanes()  # three lane formulation.

        # convert bounding boxes to nodes and build relations.
        boxes, labels, image_size = bounding_boxes
        self.get_nodes_from_bboxes(boxes, labels, coco_class_names)

        self.extract_relations()
    
    def get_nodes_from_bboxes(self, boxes, labels, coco_class_names):
        for idx, (box, label) in enumerate(zip(boxes, labels)):
            box = box.cpu().numpy().tolist()
            class_name = coco_class_names[label]

            attr = {'left': box[0], 'top': box[1], 'right': box[2], 'bottom': box[3]}
            
            # exclude vehicle dashboard
            if attr['top'] >= self.bev.params['height'] - 100: continue;
            
            if class_name not in self.actors: continue;
            else: actor_type = self.actors[class_name];

            # map center-bottom of bounding box to warped image
            x_mid = (attr['right'] + attr['left']) / 2
            y_bottom = attr['bottom']
            x_bev, y_bev = self.bev.get_projected_point(x_mid, y_bottom)

            # due to bev warp vehicles far from horizon get warped behind car, thus we will default them as far from vehcile
            if y_bev + self.bev.params['cropped_height'] > self.ego_node.attr['location_y']:
                y_bev = self.CAR_PROXIMITY_THRESH_VISIBLE / self.Y_SCALE

            #locations/distances in feet
            attr['location_x'] = x_bev * self.X_SCALE
            attr['location_y'] = y_bev * self.Y_SCALE
            attr['rel_location_x'] = self.ego_node.attr['location_x'] - attr['location_x']           # x position relative to ego
            attr['rel_location_y'] = self.ego_node.attr['location_y'] - attr['location_y']           # y position relative to ego
            attr['distance_abs'] = math.sqrt(attr['rel_location_x']**2 + attr['rel_location_y']**2)  # absolute distance from ego
            node = ObjectNode('%s_%d' % (class_name, idx), attr, actor_type)
            self.add_node(node)
            self.add_mapping_to_relative_lanes(node)




### all below can be found in relation_extractor

    def extract_relations(self):
        '''
            Extract relations between all nodes in the graph
            Builds node-wise proximity and directional relations
        '''
        for node_a, node_b in itertools.combinations(self.g.nodes, 2):
            relation_list = []
            if node_a.label == ActorType.ROAD or node_b.label == ActorType.ROAD: continue;
            if node_a.label == ActorType.CAR and node_b.label == ActorType.CAR:
                 
                if node_a.name.startswith('Ego') or node_b.name.startswith('Ego'): #starting from here its the same in relation_extractor
                    if self.get_euclidean_distance(node_a, node_b) <= self.CAR_PROXIMITY_THRESH_VISIBLE:
                        relation_list += self.extract_proximity_relations(
                            node_a, node_b)
                        relation_list += self.extract_directional_relations(
                            node_a, node_b)
                        relation_list += self.extract_proximity_relations(
                            node_b, node_a)
                        relation_list += self.extract_directional_relations(
                            node_b, node_a)
                        self.add_relations(relation_list)


    def extract_proximity_relations(self, actor1, actor2): #similar to create_proximity_relations() in rel extractor
        '''Extracts distance relations between ego node and other node'''
        if self.get_euclidean_distance(actor1, actor2) <= self.CAR_PROXIMITY_THRESH_NEAR_COLL:
            return [[actor1, Relations.near_coll, actor2]]
        elif self.get_euclidean_distance(actor1, actor2) <= self.CAR_PROXIMITY_THRESH_SUPER_NEAR:
            return [[actor1, Relations.super_near, actor2]]
        elif self.get_euclidean_distance(actor1, actor2) <= self.CAR_PROXIMITY_THRESH_VERY_NEAR:
            return [[actor1, Relations.very_near, actor2]]
        elif self.get_euclidean_distance(actor1, actor2) <= self.CAR_PROXIMITY_THRESH_NEAR:
            return [[actor1, Relations.near, actor2]]
        elif self.get_euclidean_distance(actor1, actor2) <= self.CAR_PROXIMITY_THRESH_VISIBLE:
            return [[actor1, Relations.visible, actor2]]
        return []

 
    def get_euclidean_distance(self, actor1, actor2):  #euclidean_distance() in relation extractor
        '''Absolute distance between two nodes'''
        l1 = (actor1.attr['location_x'], actor1.attr['location_y'])
        l2 = (actor2.attr['location_x'], actor2.attr['location_y'])
        return math.sqrt((l1[0] - l2[0])**2 + (l1[1] - l2[1])**2)



 
    def extract_directional_relations(self, actor1, actor2): #needs slight tweaking
        '''Extracts lateral and longitudinal relations between ego node and other node'''
        relation_list = []
        x1 = math.cos(math.radians(0)) 
        y1 = math.sin(math.radians(0))
        x2 = actor2.attr['location_x'] - actor1.attr['location_x']
        y2 = actor2.attr['location_y'] - actor1.attr['location_y']
        x2 /= math.sqrt(x2**2 + y2**2)
        y2 /= math.sqrt(x2**2 + y2**2)
 
        degree = math.degrees(math.atan2(y1, x1)) - \
                 math.degrees(math.atan2(y2, x2))
 
        degree %= 360
 
        if degree <= 45:  # actor2 is in front of actor1
            relation_list.append([actor1, Relations.atDRearOf, actor2])
        elif degree >= 45 and degree <= 90:
            relation_list.append([actor1, Relations.atSRearOf, actor2])
        elif degree >= 90 and degree <= 135:
            relation_list.append([actor1, Relations.inSFrontOf, actor2])
        elif degree >= 135 and degree <= 180:  # actor2 is behind actor1
            relation_list.append([actor1, Relations.inDFrontOf, actor2])
        elif degree >= 180 and degree <= 225:  # actor2 is behind actor1
            relation_list.append([actor1, Relations.inDFrontOf, actor2])
        elif degree >= 225 and degree <= 270:
            relation_list.append([actor1, Relations.inSFrontOf, actor2])
        elif degree >= 270 and degree <= 315:
            relation_list.append([actor1, Relations.atSRearOf, actor2])
        elif degree >= 315 and degree <= 360:
            relation_list.append([actor1, Relations.atDRearOf, actor2])
 
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


#### all below are shared btwn sg and real sg


    # TODO: move to relation_extractor in replacement of current lane-vehicle relation code
    def extract_relative_lanes(self): #needs slight tweaking
        '''
            Template: extracts vehicle lane lines {left, middle, right} relative to ego node
            Builds isIn relation between object and lane depending on x-displacement relative to the ego node
            left/middle and right/middle relations have an overlap area determined by the size of CENTER_LANE_THRESHOLD and LANE_THRESHOLD
        '''
        self.left_lane = ObjectNode('Left Lane', {}, ActorType.LANE)
        self.right_lane = ObjectNode('Right Lane', {}, ActorType.LANE)
        self.middle_lane = ObjectNode('Middle Lane', {}, ActorType.LANE)
        self.add_node(self.left_lane)
        self.add_node(self.right_lane)
        self.add_node(self.middle_lane)
        self.add_relation([self.left_lane, Relations.isIn, self.road_node])
        self.add_relation([self.right_lane, Relations.isIn, self.road_node])
        self.add_relation([self.middle_lane, Relations.isIn, self.road_node])
        self.add_relation([self.ego_node, Relations.isIn, self.middle_lane])

    def add_mapping_to_relative_lanes(self, object_node): #needs slight tweaking, called in get_nodes_from_bboxes
        '''Extracts vehicle lane line relations {left, middle, right} relative to ego node'''
        # don't build lane relations with static objects
        if object_node.label in [ActorType.LANE, ActorType.LIGHT, ActorType.SIGN, ActorType.ROAD]: return;
        if object_node.attr['rel_location_x'] < -self.LANE_THRESHOLD:
            self.add_relation([object_node, Relations.isIn, self.left_lane])
        elif object_node.attr['rel_location_x'] > self.LANE_THRESHOLD:
            self.add_relation([object_node, Relations.isIn, self.right_lane])
        if abs(object_node.attr['rel_location_x']) <= self.CENTER_LANE_THRESHOLD:
            self.add_relation([object_node, Relations.isIn, self.middle_lane])


#below are identical to those found in sg so don't need to worry about them
    def add_node(self, node):
        '''Add a single node to graph. node can be any hashable datatype including objects'''
        color = 'white'
        if 'ego' in node.name.lower():
            color = 'red'
        elif 'car' in node.name.lower():
            color = 'green'
        elif 'lane' in node.name.lower():
            color = 'yellow'
        self.g.add_node(node, attr=node.attr, label=node.name, style='filled', fillcolor=color)
 
    def add_relation(self, relation):
        '''Add relation (edge) between nodes on graph. relation is a list containing [subject, relation, object]'''
        if relation != []:
            if relation[0] in self.g.nodes and relation[2] in self.g.nodes:
                self.g.add_edge(relation[0], relation[2], object=relation[1], label=relation[1].name, color=RELATION_COLORS[int(relation[1].value)])
 
            else: raise NameError('One or both nodes in relation do not exist in graph. Relation: ' + str(relation));
 
    def add_relations(self, relations_list):
        for relation in relations_list:
            self.add_relation(relation)
 
    def visualize(self, to_filename):
        A = to_agraph(self.g)
        A.layout('dot')
        A.draw(to_filename)