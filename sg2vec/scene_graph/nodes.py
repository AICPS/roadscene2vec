class Node:
    def __init__(self, name, attr, label=None, value = None):
        self.name = name  # Car-1, Car-2.
        self.attr = attr  # bounding box info
        self.label = label  # ActorType (ie "car")
        self.value = value # ActorType index in the config's ACTOR_NAMES list 
 
    def __repr__(self):
        return "%s" % self.name
    
