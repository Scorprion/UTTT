from copy import deepcopy
import random
import torch
from torch import nn
import torch.nn.functional as F

class Node(object):
    def __init__(self, last_move):
        self.children = {}
        self.policy = []
        self.total_reward = 0
        self.visits = 0
        self.move = last_move
    
    def get_value(self):
        return self.total_reward / self.visits if self.visits > 0 else 0
    
    def is_expanded(self):
        return len(self.children) > 0
    
    def expand(self, neural_net, num_boards, all_moves, game):
        if not self.is_expanded():
            for board, board_moves in zip(num_boards, all_moves):
                for move in board_moves:
                    temp_game = game.copy()
                    self.children[board] = self.children.get(board, []) + [Node()]
    


class MCTS(object):
    def __init__(self, cpuct):
        self.cpuct = cpuct
    
    def search(self, neural_net, game_state, sims):
        # SELECT
        # EXPAND
        # UPDATE
        root = Node(None)
        current_node = root
        tree = [root]

        for n in range(sims):
            boards, moves = game_state.move_list()
            if not current_node.is_expanded():
                current_node.expand(neural_net, boards, moves, game_state)

            random.choices(weights=)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(3, 3, 3)
        self.dense1 = nn.Dense(27, 50)
        self.dense2 = nn.Dense(27, 81)
        self.dense3 = nn.Dense(27, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten after the first dim wihle preserving the batch number
        x = F.relu(self.dense1(x))
        policy = F.softmax(self.dense2(x))
        value = F.tanh(self.dense3(x))
        return value, policy

