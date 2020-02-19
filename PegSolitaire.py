from HexGrid import *
from random import randint

class PegSolitaire:
    """Game of peg solitaire"""

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

        self.board.fill_all_nodes()

        if len(empty_nodes_pos) == 0:
            nodes_pos = self.board.get_filled_nodes_positions()
            empty_node_pos = nodes_pos[randint(0,len(nodes_pos)-1)]
            self.board.clear_node(empty_node_pos)
        else: 
            for node_pos in empty_nodes_pos:
                self.board.clear_node(node_pos)

    def print_board_terminal(self):
        """ Prints the board in the terminal. For debugging purposes"""
        for row in range(self.board_size):
            row_string = ""
            for col in range(self.board_size):
                row_string += str(self.board.nodes[(row,col)]).ljust(5) if (row,col) in self.board.nodes.keys() else "".ljust(5)
            print(row_string)
    
    def show_board(self, debug=False, pause=1):
        """Show the state of the board"""
        self.board.show_graph(debug=debug, pause=pause)

    def show_board_after_action(self, action, debug=False, pause=1):
        """Show the state of the board and highlighting the remaining peg from the action"""
        self.board.show_graph(debug=debug, pause=pause, action_nodes_pos=list(action[-1:]))

    def show_board_next_action(self, action, debug=False, pause=1):
        """Show the state of the board and highlight the pegs that are going to get removed by the action"""
        self.board.show_graph(debug=debug, pause=pause,action_nodes_pos=list(action[:-1]))
    
    def get_third_node_pos(self,node_1_pos,node_2_pos):
        """Finds the third legal position in a row, given two other positions"""
        if not self.board.is_neighbors(node_1_pos, node_2_pos):
            raise Exception("The nodes with these positions are not neighbors")
        next_pos = (2*node_2_pos[0]-node_1_pos[0],2*node_2_pos[1]-node_1_pos[1])
        if next_pos in self.board.get_legal_positions():
            return next_pos

    def get_possible_actions(self):
        """Returns an array with the possible actions given its current state"""
        actions = []
        for empty_node_pos in self.board.get_empty_nodes_positions():
            for filled_node_pos in self.board.get_filled_neighbor_positions(empty_node_pos):
                third_node_pos = self.get_third_node_pos(empty_node_pos, filled_node_pos)
                if third_node_pos and self.board.get_node(third_node_pos).value:
                    actions.append((third_node_pos,filled_node_pos,empty_node_pos))
        return actions

    def perform_action(self, action):
        """Updating the state by performing the action"""
        for pos in action:
            if self.board.get_node(pos).value:
                self.board.get_node(pos).value = 0
            else:
                self.board.get_node(pos).value = 1

    def is_win(self):
        """Returns if the game is won. This is true when one peg is left"""
        return len(self.board.get_filled_nodes_positions())==1
        
    def is_lose(self):
        """Returns if the game is lost. This is true if there are several pegs left and not any possible actions"""
        return not self.is_win() and len(self.get_possible_actions())==0

    def is_done(self):
        """Returns if the game is done. This is true if the game is won or lost."""
        return self.is_lose() or self.is_win()
        
    def get_state(self):
        """Returns the state of the game"""
        return self.board.get_state()

    def set_state(self, state):
        """Sets the state of the board"""
        self.board.set_state(state)

    def get_reward(self):
        """
        Returns the award. 
        If the game is a win return 100, -1 if lose, else 0.
        """
        if self.is_win():
            return 100
        elif self.is_lose():
            return -1
        else: 
            return 0
        
    def get_end_result(self):
        """Returns how many pegs are left on the board"""
        return len(self.board.get_filled_nodes_positions())

