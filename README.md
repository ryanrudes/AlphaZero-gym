# AlphaZero-gym

An OpenAI gym environment for chess with observations and actions represented in AlphaZero-style

<a href="https://imgbb.com/"><img src="https://i.ibb.co/Fw4fhzK/Screen-Shot-2020-10-27-at-2-30-21-PM.png" alt="Screen-Shot-2020-10-27-at-2-30-21-PM" border="0"></a>

This is a modification of my `gym-chess` module. This implementation represents observations and actions using the feature planes representation method described in the AlphaZero paper, [*Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*](https://arxiv.org/pdf/1712.01815.pdf)

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
  
