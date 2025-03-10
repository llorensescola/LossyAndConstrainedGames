# Lossy-and-Constrained Extended Non-Local Games with Applications to Cryptography: BC, QKD and QPV
This is repository consists of all the code used for the paper ["Lossy-and-Constrained Extended Non-Local Games with Applications to Cryptography: BC, QKD and QPV"]([https://arxiv.org/abs/2211.06456](https://arxiv.org/abs/2405.13717)). Each of the programs corresponds to a particular case analyzed in the paper.
In the Python programs, in order to obtain the upper bounds on the optimal winning probability, the function "probwin()" has to be executed, where the argument is either the constraint condition given by $\varepsilon$ or the lossy parameter $\eta$.
## Usage
#### Install dependencies
```bash
pip3 install -r requirements.txt
```

#### Constrained BB84 monogamy-of-entanglement game (Section 4.1)
The files:
1. `constrained_BB84_epsilon_MoE_game.py` corresponds to the Python program providing the upper bounds on the optimal winning probability plotted in Figure 2.
2. `Optimization_epsilon_BB84_unentangled_strategies.nb`, `Optimization_epsilon_BB84_2qubit_strategies.nb`, and `Optimization_epsilon_BB84_MoE_strategy_3qubits.nb` correspond to the Mathematica programs providing the optimal values using unentangled, 2-qubit, and 3-qubit strategies, respectively, plotted in Figure 2.
 
####  Alice guessing game with constraints (Section 4.2)
The file `Alice_guessing_game_with_constraints.py' corresponds to the Python program used to obtain the upper vound plotted in Figure 3. The number of repetitions has to be fixed in the parameter $r$ at the begining of the code (see #Number of repetitions).

####  Local guessing game (Section 4.3)
The file `Local_guess_game.py' corresponds to the Python program used to obtain the values in Figure 4. The number of repetitions has to be fixed in the parameter $r$ at the begining of the code (see #Number of repetitions).

####  Lossy monogamy-of-entanglement games (Section 4.4)
The files:
1. `BB84_lossy_game.py' and `BB84_lossy_game_level_1AB.py'correspond to the Python programs used to obtain the values in Figure 5, using the level 1 and `1+AB', respectively.
2. `3-bases_lossy_game.py' and `3-bases_lossy_game_level_1AB.py' 'correspond to the Python programs used to obtain the values in Figure 6, using the level 1 and `1+AB', respectively.

####  Application to Bit Commitment (Section 5.1.1)
The file `Local_guess_game_accepting_loss.py' corresponds to the Python program used to obtain the values shown in Section 5.1.1. The number of thepetitions has to be set in the parameter r (see #Number of repetitions).

####  Application to Quantum Position Verificaiton (Section 5.3.1)
The file `parallel_BB84_lossy_game.py'corresponds to the Python program used to obtain the values in Figure 8. 
