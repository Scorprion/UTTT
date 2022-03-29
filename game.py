from multiprocessing.sharedctypes import Value
import numpy as np
import random

class Board(object):
    def __init__(self, board=None):
        if board is None:
            self.board = np.array([[' ' for i in range(3)] for j in range(3)])
            self.moves = list(range(9))
        else:
            self.board = np.asarray(board).reshape(3, 3)
            self.moves = np.where(self.board.flatten() == ' ')
    
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
            if(len(elems) == 1 and elems != {' '}):
                return elems.pop()

        # Check verticals
        for row in self.board.T:
            elems = set(row)
            if(len(elems) == 1 and elems != {' '}):
                return elems.pop()

        # Check diagonals
        diag = set([self.board[i, i] for i in range(3)])
        if(len(diag) == 1 and diag != {' '}):
            return elems.pop()
        
        diag = set([self.board[i, 2 - i] for i in range(3)])
        if(len(diag) == 1 and diag != {' '}):
            return elems.pop()

        # Check if no more valid moves
        if(len(self.moves) == 0):
            return '-'

        return ' '


class UTT(object):
    def __init__(self):
        self.turn = True  # X and O with X going first
        self.move_board = -1  # Board currently to be moved on - [0-8] for a specific board, otherwise -1 for any board
        self.full_board = [Board() for i in range(9)]

    def move_list(self): 
        if(self.move_board == -1): # Returns the board number and the moves for each board
            moveable_boards = [i for i, board in enumerate(self.full_board) if board.result() == ' ']
            return moveable_boards, [self.full_board[i].get_moves() for i in moveable_boards]
        else: # Returns the board number and the moves for just that board
            return [self.move_board], self.full_board[self.move_board].get_moves()

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
        if len(self.move_list()[0]) == 0:
            return '-'
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

game = UTT()
while game.result() == ' ':
    boards, moves = game.move_list()
    print(boards, moves)
    game.print_board()
    # board, move = input('Enter a move (board) (move): ').split(' ')
    # game.move(int(board), int(move))
    if isinstance(moves[0], list):
        rand_board = random.randint(0, len(boards) - 1)
        rand_move = random.choice(moves[rand_board])
        game.move(boards[rand_board], rand_move)
    else:
        game.move(random.choice(boards), random.choice(moves))
game.print_board()
print(game.result())
