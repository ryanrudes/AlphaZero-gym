# AlphaZero-gym

An OpenAI gym environment for chess with observations and actions represented in AlphaZero-style

<a href="https://imgbb.com/"><img src="https://i.ibb.co/Fw4fhzK/Screen-Shot-2020-10-27-at-2-30-21-PM.png" alt="Screen-Shot-2020-10-27-at-2-30-21-PM" border="0"></a>

This is a modification of my `gym-chess` module. This implementation represents observations and actions using the feature planes representation method described in the AlphaZero paper, [*Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*](https://arxiv.org/pdf/1712.01815.pdf)

## Installation

This requires the installation of the [`python-chess`](https://github.com/niklasf/python-chess) library. Depending on your version of installation, you may or may not have to method `chess.Board.is_repetition`. If not, the source code of this method is included in the sole Python file in this repo.

List of required packages:
* [`python-chess`](https://github.com/niklasf/python-chess)
  > `pip install chess`
* [`cairosvg`](https://pypi.org/project/CairoSVG/)
  > `pip install cairosvg`
* [`PIL`](https://pypi.org/project/Pillow/)
  > `pip install Pillow`
* [`gym`](https://github.com/openai/gym)
  > `pip install 'gym[all]'`
  
## Usage

The environment works as does any other `gym.Env` environment. Some basic functions:
* `env.reset()`
* `env.step(P)`

  > Where `P` is the policy, in the form of a probability distribution over actions represented as a matrix of shape (8, 8, 73), according to the AlphaZero method:
  > 
  > | Feature         | Planes        |
  > | :-------------- | ------------: |
  > | Queen moves     | 56            |
  > | Knight moves    | 8             |
  > | Underpromotions | 9             |
  > | Total           | 73            |
  > 
  > [Table S2: Action representation used by AlphaZero in Chess and Shogi respectively. The policy is represented by a stack of planes encoding a probability distribution over legal moves; planes correspond to the entries in the table.](https://arxiv.org/pdf/1712.01815.pdf#page=14)
  >
  > *A move in chess may be described in two parts: selecting the piece to move, and then*
  > *selecting among the legal moves for that piece. We represent the policy π(a|s) by a 8 × 8 × 73*
  > *stack of planes encoding a probability distribution over 4,672 possible moves. Each of the 8×8*
  > *positions identifies the square from which to “pick up” a piece. The first 56 planes encode*
  > *possible ‘queen moves’ for any piece: a number of squares [1..7] in which the piece will be*
  > *moved, along one of eight relative compass directions {N, NE, E, SE, S, SW, W, NW}. The*
  > *next 8 planes encode possible knight moves for that piece. The final 9 planes encode possible*
  > *underpromotions for pawn moves or captures in two possible diagonals, to knight, bishop or*
  > *rook respectively. Other pawn moves or captures from the seventh rank are promoted to a queen.*
* `env.observe()` or as an attribute, `env.state`
* `env.legal_move_mask()` (function for masking invalid actions/illegal moves in the provided action policy)
* `env.render(mode = '____')` (valid render modes are `'human'`, for visualizing the board in a pygame window, and `'rgb_array'` for returning the frame as an RGB array)
* `env.close()`

## Example

The below code sample simulates a game of chess with AlphaZero-style state-action representations:
```python
# Instantiate the environment
env = Chess()

# Reset the environment
state = env.reset()

# AlphaZero state representation shape for Chess
print (state.shape)
>> (8, 8, 119)

# Generate a random policy
p = np.random.random(size = (8, 8, 73))
# Normalize the policy
p = (p - p.min()) / (p.max() - p.min())

# Simulate a game
while not env.terminal:
  state, reward, terminal, info = env.step(p)
  env.render() # defaults to mode = 'human'
  
# Printout to demonstrate the output of the env.step() function
print(state, reward, terminal, info)

>> [[[0. 0. 0. ... 1. 1. 9.]
     [0. 1. 0. ... 1. 1. 9.]
     [0. 0. 1. ... 1. 1. 9.]
  
    ...
  
     [0. 0. 0. ... 1. 1. 9.]
     [0. 0. 0. ... 1. 1. 9.]
     [0. 0. 0. ... 1. 1. 9.]]] 0 True {'last_move': chess.Move.from_uci('c4a4'), 'turn': False}
     
# Close the environment and the display window
env.close()
```
