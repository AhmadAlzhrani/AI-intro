"""
Tic Tac Toe Player
"""

import math
import copy
from queue import Empty

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    xcount = 0
    ocount = 0

    for row in range (3):
        for col in range (3):
            if board[row][col] == X:
                xcount+=1
            if board[row][col] == O:
                ocount+=1
            
    if xcount > ocount:
        return O
    else:
        return X

    raise NotImplementedError


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """


    free = set()

    for row in range(3):
        for col in range (3):
            if board[row][col] == EMPTY:
                free.add((row,col))

    return free

    raise NotImplementedError


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """


    thePossibleMoves = actions(board)

    if action not in thePossibleMoves:
        raise Exception("This move is not possible")

    
    BoardCopy = copy.deepcopy(board)
    row = action[0]
    col = action[1]
    BoardCopy[row][col] = player(board)

    return BoardCopy

    raise NotImplementedError


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    for i in range(3):
        if board[i][0]==board[i][1]==board[i][2]:
            return board[i][0]
        if board[0][i]==board[1][i]==board[2][i]:
            return board[0][i]
    if board[1][1]==board[0][0]==board[2][2]:
        return board[1][1]
    if board[1][1]==board[0][2]==board[2][0]:
        return board[1][1]
    

    return None
    raise NotImplementedError


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True
    elif not actions(board):
        return True
    else:
        return False 

    
    raise NotImplementedError


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    if winner(board)==O:
        return -1
    elif winner(board)==X:
        return 1
    else:
        return 0
    raise NotImplementedError


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    if player(board) == O:
        moves = []
        acts = []
        for action in actions(board):
            moves.append(Max(result(board,action)))
            acts.append(action)
            index = moves.index(min(moves))
        return acts[index]
    else:
        moves = []
        acts = []
        for action in actions(board):
            moves.append(Min(result(board,action)))
            acts.append(action)
            index = moves.index(max(moves))
        return acts[index]
    raise NotImplementedError

def Max(board):
        if terminal(board):
            return utility(board)
        v = -math.inf
        optimal = None
        for action in actions(board):
            trial = Min(result(board,action))
            v = max(v,trial)
        return v
        
def Min(board):
        if terminal(board):
            return utility(board)
        v = math.inf
        optimal = None
        for action in actions(board):
            trial = Max(result(board,action))
            v = min(v,trial)
        return v