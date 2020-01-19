from HexGrid import *

class Game:

    def __init__(self,board_type, board_size, empty_nodes_pos=[]):

        if board_type not in ["Triangle","Diamond"]:
            raise Exception("Not valid board type. Either 'Triangle' or 'Diamond'")
        if board_size not in [3,4,5,6,7,8]:
            raise Exception("Not valid board size")
        if board_type=="Triangle":
            self.board = Triangle(board_size)
        else:
            self.board = Diamond(board_size)
        self.board_type = board_type
        self.board_size = board_size

        self.board.fill_all_nodes(True)
        for node_pos in empty_nodes_pos:
            self.board.get_node(node_pos).value = False

    def print_board_terminal(self):
        for row in range(self.board_size):
            row_string = ""
            for col in range(self.board_size):
                row_string += str(self.board.nodes[(row,col)]).ljust(10) if (row,col) in self.board.nodes.keys() else "".ljust(10)
            print(row_string)
    
    def show_board(self, debug=False):
        self.board.show_graph(debug=debug)
    
    def get_third_node_pos(self,node_1_pos,node_2_pos):
        if not self.board.is_neighbors(node_1_pos, node_2_pos):
            raise Exception("The nodes with these positions are not neighbors")
        next_pos = (2*node_2_pos[0]-node_1_pos[0],2*node_2_pos[1]-node_1_pos[1])
        if next_pos in self.board.get_legal_positions():
            return next_pos

    def get_possible_actions(self):
        actions = []
        for empty_node_pos in self.board.get_empty_nodes_positions():
            for filled_node_pos in self.board.get_filled_neighbor_positions(empty_node_pos):
                third_node_pos = self.get_third_node_pos(empty_node_pos, filled_node_pos)
                if third_node_pos and self.board.get_node(third_node_pos).value:
                    actions.append((third_node_pos,filled_node_pos,empty_node_pos))
        return actions

    def perform_action(self, action):
        for pos in action:
            self.board.get_node(pos).value = not self.board.get_node(pos).value

    def is_win(self):
        return len(self.board.get_filled_nodes_positions())==1
        
    def is_lose(self):
        return not self.is_win() and len(self.get_possible_actions())==0

    def is_done(self):
        return self.is_lose() or self.is_win()
        
    def get_state(self):
        return self.board.get_state()

    def set_state(self, state):
        self.board.set_state(state)


game = Game("Diamond",6,[(1,1)])
init_state = game.get_state()
actions = game.get_possible_actions()



