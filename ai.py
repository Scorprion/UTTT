from copy import deepcopy
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import datetime
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
            policy = torch.reshape(policy, (9, 9))  # NOTE: Might cause some issues. If problems arise, double check this behavior
            # total = torch.sum(torch.gather(policy, 0, torch.tensor(all_moves)))

            total = 0
            for board, board_moves in zip(num_boards, all_moves):
                for move in board_moves:
                    total += policy[board][move]

            for board, board_moves in zip(num_boards, all_moves):
                for move in board_moves:
                    self.children.append(Node(policy[board][move] / total, [board, move], not game.turn))

            self.value_update(v.detach())
            return -v.detach()
    
    def value_update(self, new_value):
        self.total_reward += new_value
        self.visits += 1


class MCTSHistory(object):
    def __init__(self):
        self.history = []

    def add_position(self, features, policy, value):
        self.history.append([features, policy, value])

    def get_batch(self, batch_size=32):
        shuffled = random.sample(self.history, len(self.history))
        for idx in range(0, len(shuffled), batch_size):
            yield shuffled[idx:idx+batch_size]


class MCTS(object):
    def __init__(self, cpuct):
        self.cpuct = cpuct
    
    def search(self, n_sims, neural_net, game_state):
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

        policy = np.zeros((9, 9))
        for child in root.children:
            policy[child.move[0], child.move[1]] = child.visits

        chosen_node = random.choices(root.children, weights=[node.visits for node in root.children])[0]
        return [*chosen_node.move], policy


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 5, 3)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(5, 5, 1)
        self.pool2 = nn.MaxPool2d(2, 1)
        self.dense1 = nn.Linear(20, 50)
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


def gen_episodes(game, neural_net, n_eps=100):
    tree = MCTS(4)
    data_tracker = MCTSHistory()
    
    for _ in range(n_eps):
        prev_info = []
        game_copy = game.copy()
        while game_copy.result() == ' ':
            [num_board, move], policy = tree.search(N_SIMS, neural_net, game_copy)
            prev_info.append([game_copy.board_features(), policy])
            game_copy.move(num_board, move)
            game_copy.print_board()

        for v, info in enumerate(prev_info[::-1]):
            data_tracker.add_position(info[0], info[1], 1 if v % 2 == 0 else -1)
        game_copy.print_board()
        print(game_copy.result())
    return data_tracker


N_SIMS = 250
brain = NeuralNet()
brain.double()
board = UTTT()
optimizer = optim.SGD(brain.parameters(), lr=1e-3, momentum=0.9)
v_loss, p_loss = nn.MSELoss(), nn.CrossEntropyLoss()

for _ in range(10):
    optimizer.zero_grad()
    data = gen_episodes(board, brain, 5)
    for batch in data.get_batch():
        X = torch.from_numpy(np.array([b[0] for b in batch]).reshape(-1, 3, 9, 9)).double()
        pred_v, pred_p = brain(X)
        real_p = torch.from_numpy(np.array([b[1] for b in batch]).reshape(-1, 81)).double()
        real_v = torch.tensor([b[2] for b in batch]).double()
        loss = v_loss(pred_v, real_v.view(-1, 1)) + p_loss(pred_p, real_p)
        print(loss)
        loss.backward()
        optimizer.step()
    torch.save(brain.state_dict(), r'C:\Users\dylan\Desktop\Code\Python\Ultimate-Tic-Tac-Toe\Models\{}.pt'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
