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
    def __init__(self, parent_node, prior, move, turn):
        self.parent = parent_node
        self.prior = prior
        self.move = move
        self.turn = turn
        self.children = []
        self.policy = []
        self.total_reward = 0
        self.visits = 0
    
    def get_value(self):
        return self.total_reward / self.visits if self.visits > 0 else 0
    
    def is_expanded(self):
        return len(self.children) > 0
    
    def expand(self, neural_net, game):
        # Check if terminal node
        outcome = game.result()
        if outcome != ' ':
            if outcome == '-':
                return 0
            else:
                return -1

        if not self.is_expanded():  # Extra layer of protection
            num_boards, all_moves = game.move_list()
            x = torch.reshape(torch.from_numpy(game.board_features()), (-1, 1, 9, 9)).float().cuda()
            v, policy = neural_net(x)
            policy = torch.reshape(torch.exp(policy), (9, 9))  # NOTE: Might cause some issues. If problems arise, double check this behavior
            # total = torch.sum(torch.gather(policy, 0, torch.tensor(all_moves)))

            total = 0
            for board, board_moves in zip(num_boards, all_moves):
                for move in board_moves:
                    total += policy[board][move]

            for board, board_moves in zip(num_boards, all_moves):
                for move in board_moves:
                    self.children.append(Node(self, policy[board][move] / total, [board, move], not game.turn))

            return v.detach()
    
    """
    Precondition: node (self) is expanded 
    """
    def max_child(self, c_puct):
        max_UCB = float('-inf')
        max_node = -1
        for child in self.children:
            ucb = child.get_value() + child.prior * c_puct * math.sqrt(self.visits) / (1 + child.visits)
            if ucb > max_UCB:
                max_UCB = ucb
                max_node = child
        return max_node
    
    def backup(self, value):
        self.total_reward += value
        self.visits += 1
        if self.parent is not None:
            self.parent.backup(-value)


class MCTSHistory(object):
    def __init__(self):
        self.history = []

    def add_position(self, features, policy, value):
        self.history.append([features, policy, value])

    def get_batch(self, batch_size=32):
        shuffled = random.sample(self.history, len(self.history))
        for idx in range(0, len(shuffled), batch_size):
            yield shuffled[idx:idx+batch_size]
        
    def merge(self, others):
        for other in others:
            self.history.extend(other)
        return self


class MCTS(object):
    def __init__(self, c_puct):
        self.c_puct = c_puct
    
    def search(self, n_sims, neural_net, game_state, training):
        root = Node(None, 0, None, game_state.turn)
        for _ in range(n_sims):
            current_node = root
            sim_board = game_state.copy()
            # SELECT
            while current_node.is_expanded():
                max_node = current_node.max_child(self.c_puct)
                sim_board.move(*max_node.move)
                current_node = max_node
                
            # EXPAND
            value = current_node.expand(neural_net, sim_board)
            
            # BACKUP
            current_node.backup(value)

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
        self.conv1 = nn.Conv2d(1, 512, 3)
        self.conv2 = nn.Conv2d(512, 512, 3)
        self.conv3 = nn.Conv2d(512, 512, 3)
        self.drop = nn.Dropout(p=0.2)
        self.dense1 = nn.Linear(4608, 500)
        self.dense2 = nn.Linear(500, 500)
        self.dense3 = nn.Linear(500, 81)
        self.dense4 = nn.Linear(500, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # Flatten after the first dim wihle preserving the batch number
        
        x = self.drop(F.relu(self.dense1(x)))
        x = self.drop(F.relu(self.dense2(x)))


        policy = F.log_softmax(self.dense3(x), 1)
        value = torch.tanh(self.dense4(x))
        return value, policy

class RunningAverage(object):
    def __init__(self):
        self.value = 0
        self.count = 0
    
    def update(self, value):
        self.value += value
        self.count += 1
    
    def get_average(self):
        return '%.2f' % (self.value / self.count if self.count != 0 else 0)

class Player(object):
    def __init__(self, n_sims, path=None):
        self.brain = NeuralNet().cuda()
        if path is not None:
            self.brain.load_state_dict(torch.load(path))
        self.n_sims = n_sims
        self.optimizer = optim.SGD(self.brain.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
        self.v_loss = nn.MSELoss()
        self.p_loss = lambda pred, real: -torch.sum(pred * real) / real.shape[0]
    
    def get_move(self, board, c_puct=1, training=True):
        tree = MCTS(c_puct)
        [num_board, move], policy = tree.search(self.n_sims, self.brain, board, training)
        return num_board, move, policy
    
    def train(self, data: MCTSHistory, batch_size, epochs):
        
        for epoch in range(epochs):
            avg_v = RunningAverage()
            avg_p = RunningAverage()

            t = tqdm(data.get_batch(batch_size), desc='Epoch #{}'.format(epoch + 1))
            for batch in t:
                X = torch.from_numpy(np.array([b[0] for b in batch]).reshape(-1, 1, 9, 9)).float().cuda()
                pred_v, pred_p = self.brain(X)
                real_p = torch.from_numpy(np.array([b[1] for b in batch]).reshape(-1, 81)).float().cuda()
                real_v = torch.tensor([b[2] for b in batch]).float().cuda()
                vloss, ploss = self.v_loss(pred_v, real_v.view(-1, 1)), self.p_loss(pred_p, real_p)

                avg_v.update(vloss.item())
                avg_p.update(ploss.item())

                t.set_postfix(loss_v=avg_v.get_average(), loss_p=avg_p.get_average())

                loss = vloss + ploss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save_brain(self, path):
        torch.save(self.brain.state_dict(), path)

    def load_brain(self, path):
        self.brain.load_state_dict(torch.load(path))
    
    def copy(self):
        return deepcopy(self)


def gen_episodes(player, game, c_puct, tracker=None, n_eps=25):
    data_tracker = MCTSHistory() if tracker is None else tracker
    
    for _ in tqdm(range(n_eps), desc='Generating Self-Play'):
        prev_info = []
        game_copy = game.copy()
        while game_copy.result() == ' ':
            num_board, move, policy = player.get_move(game_copy, c_puct)
            prev_info.append([game_copy.board_features(), policy])
            game_copy.move(num_board, move)
        if game_copy.result() == '-':
            for info in prev_info[::-1]:
                data_tracker.add_position(info[0], info[1], 0)
        else:
            for v, info in enumerate(prev_info[::-1]):
                data_tracker.add_position(info[0], info[1], 1 if v % 2 == 0 else -1)
        # game_copy.print_board()
    return data_tracker


# Pit 2 players against each other, return the player that wins over the cutoff, None otherwise
def compete(player1, player2, cutoff=0.55, matches=10):
    num_to_win = matches * cutoff
    player1_wins = 0
    player2_wins = 0
    t = tqdm(range(1, matches + 1), desc='Competing')
    for match in t:
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
        t.set_postfix({'Player1 Winrate': '%.2f' % (player1_wins / match), 'Player2 Winrate': '%.2f' % (player2_wins / match)})
    if player1_wins >= num_to_win:
        return player1
    elif player2_wins >= num_to_win:
        return player2
    return None