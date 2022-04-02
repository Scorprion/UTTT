from copy import deepcopy
import random
import torch
from torch import nn
import torch.nn.functional as F
import math

from game import UTTT

class Node(object):
    def __init__(self, prior, move, turn):
        self.children = []
        self.policy = []
        self.prior = prior
        self.total_reward = 0
        self.visits = 0
        self.move = move
        self.turn = turn
    
    def get_value(self):
        return self.total_reward / self.visits if self.visits > 0 else 0
    
    def is_expanded(self):
        return len(self.children) > 0
    
    def expand(self, neural_net, game):
        # Check if terminal node
        outcome = game.result()
        if outcome != ' ':
            game.print_board()
            if outcome == '-':
                self.value_update(0)
                return 0
            else:
                # if (self.turn and outcome == 'X') or (not self.turn and outcome == 'O')
                # the top and bottom are equivalent, just different ways of thinking; the bottom checks if the current player has to play after the game is over
                # the top checks if the current player is the winner. I just went with the bottom since it's more compact
                if self.turn is not game.turn:  
                    self.value_update(1)
                    return 1
                else:
                    self.value_update(-1)
                    return -1

        if not self.is_expanded():  # Extra layer of protection
            num_boards, all_moves = game.move_list()
            x = torch.unsqueeze(torch.from_numpy(game.board_features()), 0).double()
            v, policy = neural_net(x)
            policy = torch.reshape(policy, (9, 9))
            # total = torch.sum(torch.gather(policy, 0, torch.tensor(all_moves)))

            total = 0
            for board, board_moves in zip(num_boards, all_moves):
                for move in board_moves:
                    total += policy[board][move]

            for board, board_moves in zip(num_boards, all_moves):
                for move in board_moves:
                    self.children.append(Node(policy[board][move] / total, [board, move], not game.turn))

            self.value_update(v)
            return -v
    
    def value_update(self, new_value):
        self.total_reward += new_value
        self.visits += 1
    


class MCTS(object):
    def __init__(self, cpuct):
        self.cpuct = cpuct
    
    def search(self, neural_net, game_state, n_sims):
        root = Node(0, None, game_state.turn)
        for _ in range(n_sims):
            current_node = root
            tree = [root]
            sim_board = game_state.copy()
            # SELECT
            while current_node.is_expanded():
                max_UCB = float('-inf')
                max_node = -1
                for child in current_node.children:
                    ucb = child.get_value() + child.prior * self.cpuct * math.sqrt(current_node.visits) / (1 + child.visits)
                    if ucb > max_UCB:
                        max_UCB = ucb
                        max_node = child
                tree.append(max_node)
                sim_board.move(*max_node.move)
                current_node = max_node
                

            # EXPAND
            # if not current_node.is_expanded():
            value = current_node.expand(neural_net, sim_board)
            
            # UPDATE
            for node in tree[::-1]:
                node.value_update(value)
                value *= -1

        chosen_node = random.choices(root.children, weights=[node.visits for node in root.children])[0]
        return [*chosen_node.move]

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 27, 3)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(27, 3, 1)
        self.pool2 = nn.MaxPool2d(2, 1)
        self.dense1 = nn.Linear(12, 50)
        self.dense2 = nn.Linear(50, 81)
        self.dense3 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten after the first dim wihle preserving the batch number

        x = F.relu(self.dense1(x))
        policy = F.softmax(self.dense2(x), 1)
        value = torch.tanh(self.dense3(x))
        return value, policy


tree = MCTS(4)
brain = NeuralNet()
brain.double()
board = UTTT()
while board.result() == ' ':
    num_board, move = tree.search(brain, board, 300)
    board.move(num_board, move)
    board.print_board()

board.print_board()
print(board.result())