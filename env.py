import chess
import chess.svg

from io import BytesIO
import cairosvg
from PIL import Image

import gym
from gym.spaces import *

import numpy as np

def is_repetition(self, count: int = 3) -> bool:
    """
    Checks if the current position has repeated 3 (or a given number of)
    times.

    Unlike :func:`~chess.Board.can_claim_threefold_repetition()`,
    this does not consider a repetition that can be played on the next
    move.

    Note that checking this can be slow: In the worst case, the entire
    game has to be replayed because there is no incremental transposition
    table.
    """
    # Fast check, based on occupancy only.
    maybe_repetitions = 1
    for state in reversed(self.stack):
        if state.occupied == self.occupied:
            maybe_repetitions += 1
            if maybe_repetitions >= count:
                break
    if maybe_repetitions < count:
        return False

    # Check full replay.
    transposition_key = self._transposition_key()
    switchyard = []

    try:
        while True:
            if count <= 1:
                return True

            if len(self.move_stack) < count - 1:
                break

            move = self.pop()
            switchyard.append(move)

            if self.is_irreversible(move):
                break

            if self._transposition_key() == transposition_key:
                count -= 1
    finally:
        while switchyard:
            self.push(switchyard.pop())

    return False

chess.Board.is_repetition = is_repetition

class Chess(gym.Env):
  """AlphaGo Chess Environment"""
  metadata = {'render.modes': ['rgb_array', 'human']}

  def __init__(self):
    self.board = None

    self.T = 8
    self.M = 3
    self.L = 6

    self.size = (8, 8)

    self.viewer = None

    # self.knight_move2plane[dCol][dRow]
    """
    [ ][5][ ][3][ ]
    [7][ ][ ][ ][1]
    [ ][ ][K][ ][ ]
    [6][ ][ ][ ][0]
    [ ][4][ ][2][ ]
    """
    self.knight_move2plane = {2: {1: 0, -1: 1}, 1: {2: 2, -2: 3}, -1: {2: 4, -2: 5}, -2: {1: 6, -1: 7}}
    
    self.observation_space = Dict({"P1 piece": Tuple([MultiBinary((8, 8)) for t in range(self.T) for plane in range(6)]),
                                   "P2 piece": Tuple([MultiBinary((8, 8)) for t in range(self.T) for plane in range(6)]),
                                   "Repetitions": Tuple([MultiBinary((8, 8)) for t in range(self.T) for plane in range(2)]),
                                   "Color": MultiBinary((8, 8)),
                                   "Total move count": MultiBinary((8, 8)),
                                   "P1 castling": Tuple([MultiBinary((8, 8)) for rook in range(2)]),
                                   "P2 castling": Tuple([MultiBinary((8, 8)) for rook in range(2)]),
                                   "No-progress count": MultiBinary((8, 8))})
    
    self.action_space = Dict({"Queen moves": Tuple([MultiBinary((8, 8)) for squares in range(7) for direction in range(8)]),
                              "Knight moves": Tuple([MultiBinary((8, 8)) for move in range(8)]),
                              "Underpromotions": Tuple(MultiBinary((8, 8)) for move in range(9))})
    

  def repetitions(self):
    count = 0
    for state in reversed(self.board.stack):
      if state.occupied == self.board.occupied:
        count += 1

    return count

  def get_direction(self, fromRow, fromCol, toRow, toCol):
    if fromCol == toCol:
      return 0 if toRow < fromRow else 4
    elif fromRow == toRow:
      return 6 if toCol < fromCol else 2
    else:
      if toCol > fromCol:
        return 1 if toRow < fromRow else 3
      else:
        return 7 if toRow < fromRow else 5

  def get_diagonal(self, fromRow, fromCol, toRow, toCol):
    return int(toRow < fromRow and toCol > fromCol or toRow > fromRow and toCol < fromCol)
    
  def move_type(self, move):
    return "Knight" if self.board.piece_type_at(move.from_square) == 2 else "Queen"

  def observe(self):
    self.P1_piece_planes = np.zeros((8, 8, 6))
    self.P2_piece_planes = np.zeros((8, 8, 6))

    for pos, piece in self.board.piece_map().items():
      row, col = divmod(pos, 8)

      if piece.color == WHITE:
        self.P1_piece_planes[row, col, piece.piece_type - 1] = 1
      else:
        self.P2_piece_planes[row, col, piece.piece_type - 1] = 1

    self.Repetitions_planes = np.concatenate([np.full((8, 8, 1), int(self.board.is_repetition(repeats))) for repeats in range(1, 3)], axis = -1)
    self.Colour_plane = np.full((8, 8, 1), int(self.board.turn))
    self.Total_move_count_plane = np.full((8, 8, 1), self.board.fullmove_number)
    self.P1_castling_planes = np.concatenate((np.full((8, 8, 1), self.board.has_kingside_castling_rights(WHITE)), np.full((8, 8, 1), self.board.has_queenside_castling_rights(WHITE))), axis = -1)
    self.P2_castling_planes = np.concatenate((np.full((8, 8, 1), self.board.has_kingside_castling_rights(BLACK)), np.full((8, 8, 1), self.board.has_queenside_castling_rights(BLACK))), axis = -1)
    
    # The fifty-move rule in chess states that a player can claim a
    # draw if no capture has been made and no pawn has been moved in
    # the last fifty moves (https://en.wikipedia.org/wiki/Fifty-move_rule)
    self.No_progress_count_plane = np.full((8, 8, 1), self.board.halfmove_clock)
    
    self.binary_feature_planes = np.concatenate((self.P1_piece_planes, self.P2_piece_planes, self.Repetitions_planes), axis = -1)
    self.constant_value_planes = np.concatenate((self.Colour_plane, self.Total_move_count_plane, \
                                                 self.P1_castling_planes, self.P2_castling_planes, \
                                                 self.No_progress_count_plane), axis = -1)
    
    self.state_history = self.state_history[:, :, 14:-7]
    self.state_history = np.concatenate((self.state_history, self.binary_feature_planes, self.constant_value_planes), axis = -1)
    return self.state_history

  def reset(self):
    if self.board is None:
      self.board = chess.Board()

    self.board.reset()
    self.turn = WHITE

    self.reward = None
    self.terminal = False

    # Initialize states before timestep 1 to matrices containing all zeros
    self.state_history = np.zeros((8, 8, 14 * self.T + 7))
    return self.observe()

  def legal_move_mask(self):
    mask = np.zeros((8, 8, 73))

    for move in self.board.legal_moves:
      fromRow = 7 - move.from_square // 8
      fromCol = move.from_square % 8

      toRow = 7 - move.to_square // 8
      toCol = move.to_square % 8

      dRow = toRow - fromRow
      dCol = toCol - fromCol

      piece_type = self.board.piece_type_at(move.from_square)

      if piece_type == 2: # Knight move
        plane = knight_move2plane[dCol][dRow] + 56
      else: # Queen move
        if move.promotion and move.promotion in [2, 3, 4]: # Underpromotion move (to knight, biship, or rook)
          if fromCol == toCol: # Regular pawn promotion move
            plane = 64 + move.promotion - 2
          else: # Simultaneous diagonal pawn capture from the 7th rank and subsequent promotion
            diagonal = self.get_diagonal(fromRow, fromCol, toRow, toCol)
            plane = 64 + (diagonal + 1) * 3 + move.promotion - 2
        else: # Regular queen move
          squares = max(abs(toRow - fromRow), abs(toCol - fromCol))
          direction = self.get_direction(fromRow, fromCol, toRow, toCol)
          plane = (squares - 1) * 8 + direction

      mask[fromRow, fromCol, plane] = 1

    return mask

  def step(self, p):
    mask = self.legal_move_mask()
    p = p * mask
    pMin, pMax = p.min(), p.max()
    p = (p - pMin) / (pMax - pMin)
    action = np.unravel_index(p.argmax(), p.shape)

    fromRow, fromCol, plane = action

    if plane < 56: # Queen move
      squares, direction = divmod(plane, 8)
      squares += 1

      """
      7 0 1
      6   2
      5 4 3
      """
      if direction == 0:
        toRow = fromRow - squares
        toCol = fromCol
      elif direction == 1:
        toRow = fromRow - squares
        toCol = fromCol + squares
      elif direction == 2:
        toRow = fromRow
        toCol = fromCol + squares
      elif direction == 3:
        toRow = fromRow + squares
        toCol = fromCol + squares
      elif direction == 4:
        toRow = fromRow + squares
        toCol = fromCol 
      elif direction == 5:
        toRow = fromRow + squares
        toCol = fromCol - squares
      elif direction == 6:
        toRow = fromRow
        toCol = fromCol - squares
      else: # direction == 7
        toRow = fromRow - squares
        toCol = fromCol - squares

      fromSquare = (7 - fromRow) * 8 + fromCol
      toSquare = (7 - toRow) * 8 + toCol
      move = chess.Move(fromSquare, toSquare)
    elif plane < 64: # Knight move
      """
      [ ][5][ ][3][ ]
      [7][ ][ ][ ][1]
      [ ][ ][K][ ][ ]
      [6][ ][ ][ ][0]
      [ ][4][ ][2][ ]
      """
      if plane == 56:
        toRow = fromRow + 1
        toCol = fromCol + 2
      elif plane == 57:
        toRow = fromRow - 1
        toCol = fromCol + 2
      elif plane == 58:
        toRow = fromRow + 2
        toCol = fromCol + 1
      elif plane == 59:
        toRow = fromRow - 2
        toCol = fromCol + 1
      elif plane == 60:
        toRow = fromRow + 2
        toCol = fromCol - 1
      elif plane == 61:
        toRow = fromRow - 2
        toCol = fromCol - 1
      elif plane == 62:
        toRow = fromRow + 1
        toCol = fromCol - 2
      else: # plane == 63
        toRow = fromRow - 1
        toCol = fromCol - 2

      fromSquare = (7 - fromRow) * 8 + fromCol
      toSquare = (7 - toRow) * 8 + toCol
      move = chess.Move(fromSquare, toSquare)
    else: # Underpromotions
      toRow = fromRow - self.board.turn

      if plane <= 66:
        toCol = fromCol
        promotion = plane - 62
      elif plane <= 69:
        diagonal = 0
        promotion = plane - 65
        toCol = fromCol - self.board.turn
      else: # plane <= 72
        diagonal = 1
        promotion = plane - 68
        toCol = fromCol + self.board.turn

      fromSquare = (7 - fromRow) * 8 + fromCol
      toSquare = (7 - toRow) * 8 + toCol
      move = chess.Move(fromSquare, toSquare, promotion = promotion)

    self.board.push(move)

    # self.board = self.board.mirror()

    result = self.board.result(claim_draw = True)
    self.reward = 0 if result == '*' or result == '1/2-1/2' else 1 if result == '1-0' else -1 # if result == '0-1'
    self.terminal = self.board.is_game_over(claim_draw = True)
    self.info = {'last_move': move, 'turn': self.board.turn}

    return self.observe(), self.reward, self.terminal, self.info

  def get_image(self):
    out = BytesIO()
    bytestring = chess.svg.board(self.board, size = 256).encode('utf-8')
    cairosvg.svg2png(bytestring = bytestring, write_to = out)
    image = Image.open(out)
    return np.asarray(image)

  def render(self, mode='human'):
    img = self.get_image()

    if mode == 'rgb_array':
      return img
    elif mode == 'human':
      from gym.envs.classic_control import rendering

      if self.viewer is None:
        self.viewer = rendering.SimpleImageViewer()

      self.viewer.imshow(img)
      return self.viewer.isopen
    else:
      raise NotImplementedError()

  def close(self):
    if not self.viewer is None:
      self.viewer.close()
