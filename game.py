from multiprocessing.sharedctypes import Value
import numpy as np


class Board(object):
    def __init__(self, board=None):
        if board is None:
            self.board = np.array([['' for i in range(3)] for j in range(3)])
            self.moves = list(range(9))
        else:
            self.board = np.asarray(board)
            self.moves = np.where(self.board.flatten() == '')
    
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
            if(len(elems) == 1 and elems != {''}):
                return elems.pop()

        # Check verticals
        for row in self.board.T:
            elems = set(row)
            if(len(elems) == 1 and elems != {''}):
                return elems.pop()

        # Check diagonals
        diag = set([self.board[i, i] for i in range(3)])
        if(len(diag) == 1 and diag != {''}):
            return elems.pop()
        
        diag = set([self.board[i, 2 - i] for i in range(3)])
        if(len(diag) == 1 and diag != {''}):
            return elems.pop()

        return ''


class UTT(object):
    def __init__(self):
        self.turn = True  # X and O with X going first
        self.move_board = -1  # Board currently to be moved on - [0-8] for a specific board, otherwise -1 for any board
        self.full_board = [Board() for i in range(9)]

    def move_list(self): 
        if(self.move_board == -1): # Returns the board number and the moves for each board
            moveable_boards = [i for i, board in enumerate(self.full_board) if not board.result()]
            return moveable_boards, [self.full_board[i].get_moves() for i in moveable_boards]
        else: # Returns the board number and the moves for just that board
            return [self.move_board], self.full_board[self.move_board].get_moves()

    def move(self, board, move):
        if(self.move_board != board and self.move_board != -1):
            raise ValueError('Cannot move to that board. Invalid move')
        
        self.full_board[board].move(move, self.turn)
        self.move_board = board if not self.full_board[board].result() else -1
        self.turn = not self.turn

    def reset(self):
        self.full_board = [Board() for i in range(9)]

    # Convert the larger board to a smaller board to check for the winner
    def result(self):
        temp_board = np.empty((3, 3))
        for i, game in enumerate(self.full_board):
            temp_board[int(i / 3), i % 3] = game.result()
        temp_board = Board(temp_board)
        return temp_board.result()

    def __str__(self):
        first_row = ''
        second_row = ''
        third_row = ''
        for count, board in enumerate(self.full_board):
            first_row += board.board
            if count % 3 == 0:

test = UTT()
moveable_boards = [i for i, board in enumerate(test.full_board) if not board.result()]
print(test)