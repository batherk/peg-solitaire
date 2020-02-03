import networkx as nx
import matplotlib.pyplot as plt

class Node:

    def __init__(self, value=0):
        self.value = value
        self.neighbours = {}


    def add_neighbor(self,position,node):
        if position[0]==0 and position[1]==0:
            raise Exception("Cannot add a neighbor with same location")
        for coord in position:
            if coord not in [-1,0,1]:
                raise Exception("Not a legal position")
        if position in self.neighbours:
            raise Exception("Place already taken")

        self.neighbours[position]=node    

    def get_filled_neighbors(self):
        return [node for node in self.get_all_neighbors() if node.value]

    def get_empty_neighbors(self):
        return [node for node in self.get_all_neighbors() if not node.value]

    def get_all_neighbors(self):
        return list(self.neighbours.values())

    def is_neighbor_to(self, other_node):
        return other_node in self.get_all_neighbors()

    def get_relative_pos(self,other_node):
        for key in self.neighbours:
            if self.neighbours[key] == other_node:
                return key
        raise Exception("Nodes are not neighbors")
    

    def __str__(self):
        return str(self.value)

class HexGrid:

    def __init__(self, *args, **kwargs):
        self.nodes = {}
        self.positions = {}
        self.graph = nx.Graph()
        self.size = 0

    def get_node(self,pos):
        return self.nodes[pos]

    def fill_node(self, pos):
        self.nodes[pos].value = 1

    def clear_node(self, pos):
        self.nodes[pos].value = 0

    def get_position(self, node):
        return self.positions[node]

    def get_positions(self,nodes):
        return [self.get_position(node) for node in nodes]
    
    def get_all_nodes(self):
        return list(self.nodes.values())

    def get_filled_nodes_positions(self):
        return [node_pos for node_pos in self.nodes if self.nodes[node_pos].value]

    def get_empty_nodes_positions(self):
        return [node_pos for node_pos in self.nodes if not self.nodes[node_pos].value]

    def get_legal_positions(self):
        return list(self.nodes.keys())

    def fill_all_nodes(self):
        for node in self.get_all_nodes():
            node.value = 1

    def clear_all_nodes(self):
        for node in self.get_all_nodes():
            node.value = 0

    def add_node(self,position):
        new_node = Node()
        self.nodes[position] = new_node
        self.positions[new_node]=position
        self.graph.add_node(position)

    def add_edge(self,pos1,pos2):
        relative_pos = (pos2[0]-pos1[0],pos2[1]-pos1[1])
        self.nodes[pos1].add_neighbor(relative_pos,self.nodes[pos2])
        self.graph.add_edge(pos1,pos2)

    def show_graph(self, positions=None, debug=False):
        if not positions:
            positions = {}
            labels = {}
            label_positions = {}
            for pos in self.get_legal_positions():
                positions[pos]=(pos[1],self.size - pos[0])
                labels[pos]=str(pos)
                label_positions[pos] = (pos[1],self.size - pos[0]-0.3)
        
        nx.draw_networkx_nodes(self.graph, positions, nodelist=self.get_empty_nodes_positions(), node_color='black')
        nx.draw_networkx_nodes(self.graph, positions, nodelist=self.get_filled_nodes_positions(), node_color='blue')
        nx.draw_networkx_edges(self.graph, positions, alpha=0.5, width=1)

        if debug:
            nx.draw_networkx_labels(self.graph, label_positions,labels=labels)
        plt.axis('off')
        plt.show()

    def is_neighbors(self, node1_pos, node2_pos):
        return self.get_node(node1_pos).is_neighbor_to(self.get_node(node2_pos))

    def get_filled_neighbor_positions(self,node_pos):
        return self.get_positions(self.get_node(node_pos).get_filled_neighbors())

    def get_state(self):
        state = ""
        for pos in self.nodes:
            state += str(self.nodes[pos].value)
        return state

    def set_state(self, state):
        index = 0
        for pos in self.nodes:
            self.nodes[pos].value = int(state[index])
            index += 1

class Diamond(HexGrid):
    
    def __init__(self,size):
        super(Diamond, self).__init__()
        self.size = size

        for i in range(size):
            for j in range(size):
                self.add_node((i,j))

        for pos in self.get_legal_positions():
            for relative_pos in [(-1,0),(1,0),(0,1),(0,-1),(1,-1),(-1,1)]:
                neighbor_pos = (pos[0]+relative_pos[0],pos[1] + relative_pos[1])
                if neighbor_pos in self.nodes:
                    self.add_edge(pos,neighbor_pos)

    def show_graph(self, debug=False):
        if debug:
            super(Diamond, self).show_graph(debug=True)
        else: 
            positions = {}
            for pos in self.get_legal_positions():
                x = pos[1]-pos[0]
                y = 2*self.size - pos[0] - pos[1]
                positions[pos]=(x,y)
            super(Diamond, self).show_graph(positions=positions)   

class Triangle(HexGrid):
    
    def __init__(self,size):
        super(Triangle, self).__init__()
        self.size = size

        for i in range(size):
            for j in range(i+1):
                self.add_node((i,j))

        for pos in self.get_legal_positions():
            for relative_pos in [(-1,0),(1,0),(0,1),(0,-1),(-1,-1),(1,1)]:
                neighbor_pos = (pos[0]+relative_pos[0],pos[1] + relative_pos[1])
                if neighbor_pos in self.nodes:
                    self.add_edge(pos,neighbor_pos)

    def show_graph(self, debug=False):
            if debug:
                super(Triangle, self).show_graph(debug=True)
            else: 
                positions = {}
                for pos in self.get_legal_positions():
                    positions[pos]=(2*pos[1]-pos[0],self.size - pos[0])
                super(Triangle, self).show_graph(positions)
