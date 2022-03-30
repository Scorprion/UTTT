from copy import deepcopy
import random

class Node(object):
    def __init__(self, board):
        self.reward = 0
        self.visits = 0
        self.
    


class MCTS(object):
    def __init__(self, root: Node):
        self.tree = root
    
    def search(self, game_state, sims):
        for n in range(sims):
            boards, moves = game_state.move_list()
            
            random.choices(weights=)

