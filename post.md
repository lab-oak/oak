[HEADING=1]Introducing:[/HEADING]
Oak
https://github.com/lab-oak/oak/
is a high performance software toolkit principally for training battle and team-building neural networks in RBY.


The scope of the library and programs is large. It aims to catch up Pokemon AI with developments that have been very successful in Chess.


[HEADING=1]Context[/HEADING]
The central fact in any AI development is that Pokemon is an imperfect information game. In this regime the direct analog of tree search is called Counterfactual Regret Minimization. The reality is that CRM is basically completely infeasible. It entails solving the entire game tree and so it's only been applied to small variants of Poker; RBY OU is way too large. Modifications to the algorithm that make it work for large games (e.g. ReBeL/Student of Games) are enormously complex and expensive.


This is not fatal to our effort to create superhuman agents. It just means that a provable, out-of-the-box solution is not available.


The simplest way to attack imperfect information is actually the most reliably effective. [I]Information Set MCTS[/I] starts with 'determinization', where we fill-in the hidden information. Then we just perform a vanilla Monte-Carlo tree search on this now perfect info game. This two step process is done multiple times in parallel, and the value and policies from the many searches are combined.


With this approach, we claim we can split the problem into two independent sub-problems: accurately predicting the opponents team and strong tree search on a [I]simultaneous move[/I] and [I]stochastic[/I], but still perfect info game.


This is the approach that this project supports, with a (currently) sole focus on the latter problem.


[HEADING=1]Features[/HEADING]
Now that the game is perfect info, we can take advantage of the scalable and general purpose approaches that worked for Chess and Go and various other games. Below is what Oak provides in that respect.


[HEADING=3]Fast Simulator[/HEADING]
Most projects use Pokemon-Showdown as the base simulator since it is open source and the de facto implementation of the game. Unfortunately they are doomed from the start since PS is orders of magnitude too slow for anything serious.
This project uses `libpkmn` which 1000x faster than PS and also matches its behaviour exactly. Depending on the eval, we are able to perform a million mcts iterations in only a few seconds.
Oak implements 


[HEADING=3]Proper Search[/HEADING]
It is straight-forward to modify MCTS for Pokemon. Just make each node store data for both players and the vanilla selection and update phases are executed jointly. This can be empricially effective but it is unsound. It does not produce low exploitability strategies no matter the number of iterations that are performed.
It is also known there is a fix. This project implements Exp3, which is an 'adversarial' algorithm unlike the typical UCB. These algorithms provably converge to equilibrium. In any case, we also provide both algorithms, together with variants that also take advantage of a neural network's policy inference.


[HEADING=3]Complex Eval[/HEADING]
It is very easy to create a decent value estimator for chess: an eval that simply totals the piece values for both players will beat most amateurs. If invoked millions of times in a search tree, it will very effectively avoid blunders and be very good at tactics. The first chess superhuman engines did basically this, they evaluated positions using a weighted sum of material and strategic scores.


These engines were surpassed by neural networks which are capable of much more complex estimation and are not polluted by human bias. At first these networks were large enough to require a GPU but eventually it was discovered that smaller, faster networks running on CPU were even more effective.


Oak implements small battle networks that perform hundreds of thousands of inferences per second. There is no 'best practice' for Pokemon NNs so we have what is essentially a MLP that takes a certain one-hot tabular encoding of the battle as input and outputs a number [0, 1] for the value estimate. It has an optional 'policy head' for move prediction.
[HEADING=3]Fast Data Generation[/HEADING]

TODO


[HEADING=3]Extensible Data formats[/HEADING]

TODO


[HEADING=3]Scripting Support[/HEADING]

TODO


[HEADING=1]Results[/HEADING]
My compute is limited to an old laptop and I have little enthusiasm for *training* networks. For this reason I don't mind acknowleging that my successes with this program have been modest.


[HEADING=3]Battling[/HEADING]
The basic idea to train a network is to iterate: start with some baseline agent (in this case Monte Carlo eval) and use self-play to generate training data. Use the data to train a network stronger then that agent, and repeat.


With the default params I was consitently able to train a network that outperformed the monte-carlo agent that generated the data. This is not ground breaking because monte carlo eval is quite weak, but it also should be mentioned that, in my testing, monte carlo beats FoulPlay's and my own hand crafted eval.


[HEADING=3]Building[/HEADING]
First, I should mention that the team building aspect is not the focus and its potential is limited. This is because it generates teams using a simple policy rollout, no search is used. 


[HEADING=1]Intent[/HEADING]


The ideas and methods here were tested and proven by really by the two historically strongest engines. Stockfish and LeelaChess0 are large, public efforts. The code is developed my a small team of programmers. Data is generated and networks are trained and evaluated by many volunteers.


I would like to see something like this for Pokemon. I believe Pokemon is a very hard game to attack and collaboration and organization are necessary.