class Node:
    def __init__(self, name, attr, type=None):
        self.name = name
        self.attr = attr
        self.label = name
        self.type = type.value if type != None else None

    def __repr__(self):
        return "%s" % self.name
    
class ObjectNode:
    def __init__(self, name, attr, label):
        self.name = name  # Car-1, Car-2.
        self.attr = attr  # bounding box info
        self.label = label  # ActorType

    def __repr__(self):
        return '%s' % (self.name)