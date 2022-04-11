from ai import *
import multiprocessing as mp

# TODO: Add multiprocessing to speed up the sims and competition
# TODO: Fix neural net updating strategy (only wipe data when nn is updated, otherwise keep the current data and searched tree(?))'
# TODO: Reconsider neural network architecture and learning rate (perhaps add some batch norm and change the linear part)
# TODO: Keep tree when not replaced
# HYPERPARAMS
SIMS = 25
EPISODES = 50  # 4 minutes for 50 episodes
EPOCHS = 10
BATCH_SIZE = 64
MATCHES = 40
THRESHOLD = 0.55
C_PUCT = 1

prev_best = Player(n_sims=SIMS, path=r'C:\Users\dylan\Desktop\Code\Python\Ultimate-Tic-Tac-Toe\Models\20220411121222.pt')
current_player = prev_best.copy() # Player(n_sims=100)
board = UTTT()
data = MCTSHistory()

while True:
    data = gen_episodes(current_player, board, C_PUCT, data, EPISODES)
    current_player.train(data, batch_size=BATCH_SIZE, epochs=EPOCHS)

    current_player.brain.eval()
    prev_best.brain.eval()
    result = compete(prev_best, current_player, cutoff=THRESHOLD, matches=MATCHES)
    current_player.brain.train()
    prev_best.brain.train()

    if result == current_player:
        print('New Winner!')
        current_player.save_brain(path=r'C:\Users\dylan\Desktop\Code\Python\Ultimate-Tic-Tac-Toe\Models\{}.pt'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
        prev_best = current_player.copy()
        data = None
    else:
        # If the current player didn't beat the previous best, then reset to that state while keeping the stored data
        current_player = prev_best.copy()

"""
game = UTTT()
while game.result() == ' ':
    boards, moves = game.move_list()
    # print(boards, moves)
    # game.print_board()
    
    board, move = input('Enter a move (board) (move): ').split(' ')
    game.move(int(board), int(move))
    
    
    rand_board = random.randint(0, len(boards) - 1)
    rand_move = random.choice(moves[rand_board])
    game.move(boards[rand_board], rand_move)
    game.print_board()
    print(game.board_features())

game.print_board()
print(game.result())
"""