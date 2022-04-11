from multiprocessing.sharedctypes import Value
from copy import deepcopy
import numpy as np
import random

class Board(object):
    def __init__(self, board=None):
        if board is None:
            self.board = np.array([[' ' for i in range(3)] for j in range(3)])
            self.moves = list(range(9))
        else:
            self.board = np.asarray(board).reshape(3, 3)
            self.moves = np.where(self.board.flatten() == ' ')[0]
            # print(self.board, self.result())
    
    def get_moves(self):
        return self.moves
    
    def move(self, position: int, turn: bool):
        if(position not in self.moves):
            raise ValueError('Invalid move {}'.format(position))

        self.board[int(position / 3)][position % 3] = 'X' if turn else 'O'
        self.moves.remove(position)
        return self.result()

    def __str__(self):
        return str(self.board)
    
    def result(self):
        # Check horizontals
        for row in self.board:
            elems = set(row)
            if(len(elems) == 1 and not elems.issubset({' ', '-'})):
                return elems.pop()

        # Check verticals
        for row in self.board.T:
            elems = set(row)
            if(len(elems) == 1 and not elems.issubset({' ', '-'})):
                return elems.pop()

        # Check diagonals
        diag = set([self.board[i, i] for i in range(3)])
        if(len(diag) == 1 and not diag.issubset({' ', '-'})):
            return diag.pop()
        
        diag = set([self.board[i, 2 - i] for i in range(3)])
        if(len(diag) == 1 and not diag.issubset({' ', '-'})):
            return diag.pop()

        # Check if no more valid moves
        if(len(self.moves) == 0):
            return '-'

        return ' '


class UTTT(object):
    def __init__(self, history=6):
        self.turn = True  # X and O with X going first
        self.move_board = -1  # Board currently to be moved on - [0-8] for a specific board, otherwise -1 for any board
        self.full_board = [Board() for i in range(9)]
        self.features = []

    def move_list(self): 
        if(self.move_board == -1): # Returns the board number and the moves for each board
            moveable_boards = [i for i, board in enumerate(self.full_board) if board.result() == ' ']
            return moveable_boards, [self.full_board[i].get_moves() for i in moveable_boards]
        else: # Returns the board number and the moves for just that board
            return [self.move_board], [self.full_board[self.move_board].get_moves()]

    def move(self, board, move):
        if(self.move_board != board and self.move_board != -1):
            raise ValueError('Cannot move to that board. Invalid move')
        
        self.full_board[board].move(move, self.turn)
        self.move_board = move if self.full_board[move].result() == ' ' else -1
        self.turn = not self.turn

    def reset(self):
        self.full_board = [Board() for i in range(9)]

    # Convert the larger board to a smaller board to check for the winner
    def result(self):
        temp_board = np.array([[' ' for i in range(3)] for j in range(3)])
        for i, game in enumerate(self.full_board):
            temp_board[int(i / 3), i % 3] = game.result()
        temp_board = Board(temp_board)
        
        return temp_board.result()

    def print_board(self):
        first_row = ''
        second_row = ''
        third_row = ''
        for count, board in enumerate(self.full_board):
            first_row += '|' + ' '.join(board.board[0]) + '|'
            second_row += '|' + ' '.join(board.board[1]) + '|'
            third_row += '|' + ' '.join(board.board[2]) + '|'
            if (count + 1) % 3 == 0:
                print(first_row)
                print(second_row)
                print(third_row)
                print('---------------------')
                first_row = ''
                second_row = ''
                third_row = ''
    
    def copy(self):
        return deepcopy(self)

    def board_features(self):
        # 1 board for each player, switched based on turn, and another layer filled with an integer corresponding to whose turn it is
        row_boards = []
        total_boards = []
        for board_num, square in enumerate(self.full_board):
            sub_board = np.zeros((3, 3))
            sub_board[np.where(square.board == 'X')] = 1
            sub_board[np.where(square.board == 'O')] = -1
            row_boards.append(sub_board)
            
            if (board_num + 1) % 3 == 0:
                total_boards.append(np.hstack((np.asarray(row_boards))))
                row_boards = []
        total_boards = np.vstack(np.asarray(total_boards))
        return total_boards if self.turn else -total_boards
