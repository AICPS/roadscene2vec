import matplotlib, math, itertools
matplotlib.use("Agg")
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
from .relation_extractor import Relations, ActorType, RELATION_COLORS 
from .nodes import Node
from .nodes import ObjectNode
import pdb

'''Create scenegraph using raw Carla json frame data or raw image data'''
class SceneGraph:
    
    #graph can be initialized with a framedict containing raw Carla data to load all objects at once
    def __init__(self, relation_extractor, framedict= None, framenum=None, bounding_boxes = None, bev = None, coco_class_names=None, platform='carla'):
        #configure relation extraction settings
        self.relation_extractor = relation_extractor
        
#         pdb.set_trace()
        
        self.platform = platform
        
        if self.platform == "carla":
            self.g = nx.MultiDiGraph() #initialize scenegraph as networkx graph
            self.road_node = Node("Root Road", {}, ActorType.ROAD)
            self.add_node(self.road_node)   #adding the road as the root node
            self.parse_json(framedict) # processing json framedict
        elif self.platform == "image":
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
            

            self.g = nx.MultiDiGraph()  # initialize scenegraph as networkx graph
            # road and lane settings.
            # we need to define the type of node.
            self.road_node = ObjectNode('Root Road', {}, ActorType.ROAD)
            self.add_node(self.road_node)   # adding the road as the root node
    
            # set ego location to middle-bottom of image.
            # set ego location to middle-bottom of image.
            self.ego_location = bev.get_projected_point(
                                    bev.params['width']/2, 
                                    bev.params['height'])

            self.ego_location = bev.apply_depth_estimation(
                                    self.ego_location[0], 
                                    self.ego_location[1])
            
            self.egoNode = ObjectNode('ego car', {

                                       'location_x': self.ego_location[0], 
                                       'location_y': self.ego_location[1]}, 
                                       ActorType.CAR)
    
            # add ego-vehicle to graph
            self.add_node(self.egoNode)
            
            # add middle, right, and left lanes to graph
            self.relation_extractor.extract_relative_lanes(self) 
    
            # convert bounding boxes to nodes and build relations.
            boxes, labels, image_size = bounding_boxes
            self.get_nodes_from_bboxes(bev, boxes, labels, coco_class_names)

            # add all node-pairwise relations
            #perhaps add below to rel extractor
            for node_a, node_b in itertools.combinations(self.g.nodes, 2):
                if node_a != node_b:
                    if node_a.label == ActorType.ROAD or node_b.label == ActorType.ROAD: continue;
                    if node_a.label == ActorType.CAR and node_b.label == ActorType.CAR:
                        relation_list = self.relation_extractor.extract_relations_car_car(node_a, node_b) #only want to build car car rels for real img sg?
                        self.add_relations(relation_list)


    def get_nodes_from_bboxes(self, bev, boxes, labels, coco_class_names):
        for idx, (box, label) in enumerate(zip(boxes, labels)):
            box = box.cpu().numpy().tolist()
            class_name = coco_class_names[label]

            attr = {'left': box[0], 'top': box[1], 'right': box[2], 'bottom': box[3]}
            
            # exclude vehicle dashboard
            if attr['top'] >= bev.params['height'] - 100: continue;
            
            # filter traffic participants
            if class_name not in self.actors: continue;
            else: actor_type = self.actors[class_name];

            # map center-bottom of bounding box to warped image
            x_mid = (attr['right'] + attr['left']) / 2
            y_bottom = attr['bottom']
            x_bev, y_bev = bev.get_projected_point(x_mid, y_bottom)

            # approximate locations / distances in feet
            attr['location_x'], attr['location_y'] = bev.apply_depth_estimation(x_bev, y_bev)

            # due to bev warp, vehicles far from horizon get warped behind car, thus we will default them as far from vehcile
            if attr['location_y'] > self.egoNode.attr['location_y']:
                # should store this in a list dictating the filename of the scene
                print('BEV warped to behind vehcile')
                attr['location_y'] = self.egoNode.attr['location_y'] - self.relation_extractor.CAR_PROXIMITY_THRESH_VISIBLE 

            attr['rel_location_x'] = attr['location_x'] - self.egoNode.attr['location_x']           # x position relative to ego (neg left, pos right)
            attr['rel_location_y'] = attr['location_y'] - self.egoNode.attr['location_y']           # y position relative to ego (neg vehicle ahead of ego)
            attr['distance_abs'] = math.sqrt(attr['rel_location_x']**2 + attr['rel_location_y']**2) # absolute distance from ego
            node = ObjectNode('%s_%d' % (class_name, idx), attr, actor_type)
            
            # add vehicle to graph
            self.add_node(node)

            # add lane vehicle relations to graph
            self.relation_extractor.add_mapping_to_relative_lanes(self, node)


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


# add all pair-wise relations between two nodes
    def add_relations(self, relations_list):
        for relation in relations_list:
            self.add_relation(relation)
    

    # add a single pair-wise relation between two nodes
    def add_relation(self, relation):
        if relation != []:
            node1, edge, node2 = relation
            if node1 in self.g.nodes and node2 in self.g.nodes:
                self.g.add_edge(node1, node2, object=edge, label=edge.name, color=RELATION_COLORS[int(edge.value)])
            else:
                raise NameError("One or both nodes in relation do not exist in graph. Relation: " + str(relation))
            

    #parses actor dict and adds nodes to graph. this can be used for all actor types.
    def add_actor_dict(self, actordict):
        for actor_id, attr in actordict.items():
            # filter actors behind ego 
            x1, y1 = math.cos(math.radians(self.egoNode.attr['rotation'][0])), math.sin(math.radians(self.egoNode.attr['rotation'][0]))
            x2, y2 = attr['location'][0] - self.egoNode.attr['location'][0], attr['location'][1] - self.egoNode.attr['location'][1]
            inner_product = x1*x2 + y1*y2
            length_product = math.sqrt(x1**2+y1**2) + math.sqrt(x2**2+y2**2)
            degree = math.degrees(math.acos(inner_product / length_product))
            
            if degree <= 80 or (degree >=280 and degree <= 360):
                # if abs(self.egoNode.attr['lane_idx'] - attr['lane_idx']) <= 1 \
                # or ("invading_lane" in self.egoNode.attr and (2*self.egoNode.attr['invading_lane'] - self.egoNode.attr['orig_lane_idx']) == attr['lane_idx']):
                n = Node(actor_id, attr, None)   #using the actor key as the node name and the dict as its attributes.
                n.name = self.relation_extractor.get_actor_type(n).name.lower() + ":" + actor_id
                n.type = self.relation_extractor.get_actor_type(n).value
                self.add_node(n)
                self.relation_extractor.add_mapping_to_relative_lanes(self, n)
            

    #adds lanes and their dicts. constructs relation between each lane and the root road node.
    def add_lane_dict(self, lanedict):
        #TODO: can we filter out the lane that has no car on it?
        for idx, lane in enumerate(lanedict['lanes']):
            lane['lane_idx'] = idx
            n = Node("lane:"+str(idx), lane, ActorType.LANE)
            self.add_node(n)
            self.add_relation([n, Relations.isIn, self.road_node])
            

    #add signs as entities of the road.
    def add_sign_dict(self, signdict):
        for sign_id, signattr in signdict.items():
            n = Node(sign_id, signattr, ActorType.SIGN)
            self.add_node(n)
            self.add_relation([n, Relations.isIn, self.road_node])


    #add the contents of a whole framedict to the graph
    def parse_json(self, framedict):
        
        self.egoNode = Node("ego:"+framedict['ego']['name'], framedict['ego'], ActorType.CAR)
        self.add_node(self.egoNode)

        #rotating axes to align with ego. yaw axis is the primary rotation axis in vehicles
        self.ego_yaw = math.radians(self.egoNode.attr['rotation'][0])
        self.ego_cos_term = math.cos(self.ego_yaw)
        self.ego_sin_term = math.sin(self.ego_yaw)
        self.relation_extractor.extract_relative_lanes(self)

#         self.relation_extractor = RelationExtractor(self.egoNode) #see line 99
        for key, attrs in framedict.items():   
            # if key == "lane":
            #     self.add_lane_dict(attrs)get_euclidean_distance
            if key == "sign":
                self.add_sign_dict(attrs)
            elif key == "actors":
                self.add_actor_dict(attrs)
        self.relation_extractor.extract_semantic_relations(self)
        

    def visualize(self, filename=None):
        A = to_agraph(self.g)
        A.layout('dot')
        A.draw(filename)