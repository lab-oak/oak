# Stockfish for RBY

This project deserves some context. It would be nice to first describe, as quickly as I can, the state of the art of computer search in two different games.

## Chess

Chess is king, baby. It was the the motivator for basically every invention in computer search. 
AlphaBeta/Monte Carlo Tree Search (MCTS) were developed and optimized specifically to attack this game, and many other techniques and wisdoms were made too.

By the late 90s computer engines were unquestionaly stronger than the best players. This was achieved by writing superfast AlphaBeta implementations and finely tuning a crude value estimation function.
Chess and AB are a match made in heaven, and this is largely responsible for superhuman strength. The value estimation was a fast hueristic that basically added up piece values and various stretegic scores.

This approach plateaud for a while and it was understood that you could never beat a computer but also that engines could not understand certain quiet positions. They could not understand long term strategy.

AlphaZero and later neural network searches fixed this problem and overtook conventional engines.
They took the opposite approach: slower search but a more complex value estimation.
Today the best engines combine these approaches with small but powerful neural networks that run on CPU instead of GPU. The best players can't beat their phones.

## Pokemon

Picture a sad Durant holding a bindle.

Pokemon doesnt have many examples what could be called an engine. The majority of projects don't use search at all; They give you a probabiliy distribution over your legal moves. This is a fundamentally weaker tool than a search engine that can run indefinitely. They have been limited to "top X on Y ladder".

Pokemon is just much more difficult than Chess. It doesn't have a love like Chess and AB's. In fact, alpha beta is fatally flawed on simultaneious move games; It's worst case performance is a disaster. This means thats Monte Carlo Tree Search (MCTS) is the preferred approach. 

The fact that pokemon is simultaneous move is horrible for many other reasons. For starters it basically freakin squares the branching factor of the game! It also spoils the theoretical results that might power the newer approaches like Monte Carlo Tree Search (MCTS).

Getting data for a strong neural network is very hard and expensive. For context more than a *billion* expert examples may be used when training the tiny network that stockfish relies on. There aren't enough public high ladder games going on, for starters.

Pokemon-Showdown, the first choice of anyone who has ever attempted this problem, is a trap. Its architecture is matter of factly too slow. Its so slow that it wouldn't even matter if all the other problems went away. This means that computer generated data is very expensive too. Coders who realize this and continue on write their own simulators, which introduces errors from differences in the simulators' mechanics.

Recently however some hope has appeared.

There exists a battle simulator that is so fast that it is basically perfect. Like it probably won't ever be outdone or seriously improved, and it crucially also matches showdown exactly. The catch is that it currently only supporst gen 1... which honestly was kinda 😬 at first but now I embrace it. The simplest generation should be targeted first. But seriously if nothing else comes from this post let it be more challengers using Libpkmn instead of Showdown.

The theoretical problems have been solved too. Slight changes to the usual MCTS approaches restore the 'guarantees'. It remains to be seen if the modified approaches will be as effective.

In light of this, there is a path forward to real progress.

## Plan of Attack

I won't waste time explaining how how crucial information management is in Pokemon. Its yet another thing that makes this problem so difficult. I will though try to convince you that its not that bad, and the first step to dealing with it is to ignore it.

Naturally, the proofs which say perfect play is possible if you just run the search long enough require that the game is perfect information. In pokemon, this just means both players have `!showteam'd`.

This is not a ridiculous assumption. First of all, hidden information can only shrink over the course of the game. In some games it shrinks fast since good players can make educated guesses about unrevealed moves and items. The only other unknowns are the EVs, which are accurately deduced by damage rolls (this is actually one thing some projects do very well). Also RBY doesn't have EVs or items, only species and moves are hidden.

The most generically successful approach for dealing with imperfect info games is Information-Set MCTS. The game state is determinized (i.e. we guess the unrevealed species/moves of the opponent) and then run a perfect info search from there. Typically the state is determinized many times and searches are run in parallel. The 

This is also 

There is at least one promising starter, FoulPlay, which has topped OU/Randbats ladders. It is worthy of praise and emulation but I have to also admit that I don't think there is a case that it is competitive with the best players. It also uses an extrememly simple evaluation function.

# Introducing Oak

I hope the previous section was enough to explain how whatever. I hope this section can convince you that 

Oak is a code library and a collection of programs that tries to address all the problems I outlined.

- Everything is fast. The base simulator 



 The scope of what Oak do is vast and I've already written a tutorial and documentatino on the github repo.

- There project is available as source code only. It is basically trivial to build on up-to-date Linux machines

- I am poo and pee
