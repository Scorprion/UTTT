from copy import deepcopy
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import datetime
import math
from tqdm import tqdm

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
            policy = torch.reshape(torch.exp(policy), (9, 9))  # NOTE: Might cause some issues. If problems arise, double check this behavior
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
    
    def search(self, n_sims, neural_net, game_state, training):
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
        policy /= policy.sum()

        if training:
            chosen_node = random.choices(root.children, weights=[node.visits for node in root.children])[0]
        else:
            chosen_node = max(root.children, key=lambda node: node.visits)
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
        policy = F.log_softmax(self.dense2(x), 1)
        value = torch.tanh(self.dense3(x))
        return value, policy


class Player(object):
    def __init__(self, n_sims=250, path=None):
        self.brain = NeuralNet()
        if path is not None:
            self.brain.load_state_dict(torch.load(path))
        self.brain.double()
        self.n_sims = n_sims
        self.optimizer = optim.Adam(self.brain.parameters())
        self.v_loss = nn.MSELoss()
        self.p_loss = lambda pred, real: -torch.sum(pred * real) / real.shape[0]
    
    def get_move(self, board, training=True):
        tree = MCTS(4)
        [num_board, move], policy = tree.search(self.n_sims, self.brain, board, training)
        return num_board, move, policy
    
    def train(self, data: MCTSHistory, batch_size, epochs):
        self.optimizer.zero_grad()
        for i in range(epochs):
            for batch in data.get_batch(batch_size):
                X = torch.from_numpy(np.array([b[0] for b in batch]).reshape(-1, 3, 9, 9)).double()
                pred_v, pred_p = self.brain(X)
                real_p = torch.from_numpy(np.array([b[1] for b in batch]).reshape(-1, 81)).double()
                real_v = torch.tensor([b[2] for b in batch]).double()
                vloss, ploss = self.v_loss(pred_v, real_v.view(-1, 1)), self.p_loss(pred_p, real_p)
                print(vloss, ploss)
                loss = vloss + ploss
                loss.backward()
                self.optimizer.step()

    def save_brain(self, path):
        torch.save(self.brain.state_dict(), path)

    def load_brain(self, path):
        self.brain.load_state_dict(torch.load(path))
    
    def copy(self):
        return deepcopy(self)


def gen_episodes(player, game, n_eps=25):
    data_tracker = MCTSHistory()
    
    for _ in range(n_eps):
        prev_info = []
        game_copy = game.copy()
        while game_copy.result() == ' ':
            num_board, move, policy = player.get_move(game_copy)
            prev_info.append([game_copy.board_features(), policy])
            game_copy.move(num_board, move)
            print(num_board, move)

        for v, info in enumerate(prev_info[::-1]):
            data_tracker.add_position(info[0], info[1], 1 if v % 2 == 0 else -1)
        game_copy.print_board()
        print(game_copy.result())
    return data_tracker


# Pit 2 players against each other, return the player that wins over the cutoff, None otherwise
def compete(player1, player2, cutoff=0.55, matches=10):
    num_to_win = matches * cutoff
    player1_wins = 0
    player2_wins = 0
    for _ in range(matches):
        p1 = player1
        p2 = player2
        game = UTTT()
        if random.choice([True, False]):  # 50 / 50 chance that p2 plays first
            p1 = player2
            p2 = player1
        
        while game.result() == ' ':
            num_board, move, p = p1.get_move(game, training=False)
            game.move(num_board, move)
            if game.result() != ' ':
                break
            num_board, move, p = p2.get_move(game, training=False)
            game.move(num_board, move)
        
        if game.result() == 'X':
            if p1 == player1:
                player1_wins += 1
            else:
                player2_wins += 1
        elif game.result() == 'O':
            if p1 == player1:
                player2_wins += 1
            else:
                player1_wins += 1
    if player1_wins >= num_to_win:
        return player1
    elif player2_wins >= num_to_win:
        return player2
    return None



prev_best = Player(n_sims=75)  # , path=r'C:\Users\dylan\Desktop\Code\Python\Ultimate-Tic-Tac-Toe\Models\20220403015231.pt'
current_player = prev_best.copy() # Player(n_sims=100)
board = UTTT()

while True:
    data = gen_episodes(current_player, board, 25)
    current_player.train(data, batch_size=16, epochs=3)
    result = compete(prev_best, current_player, cutoff=0.6, matches=10)
    if result == current_player:
        print('New Winner!')
        current_player.save_brain(path=r'C:\Users\dylan\Desktop\Code\Python\Ultimate-Tic-Tac-Toe\Models\{}.pt'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
        prev_best = current_player.copy()
