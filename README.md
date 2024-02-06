# In-distribution_breakout

# Breakout 

This subdirectory contains the source code of sanitization backdoor policies for Atari breakout game environment. The backdoor policy in this example has been trained using the environment poisoning framework of TrojDRL [paper](https://arxiv.org/pdf/1903.06638.pdf) .

The state space consists of a concatenated image frames. The trigger is a "missing-tile" shaped image inserted on the tile space of the game. The in-distribution backdoor policy has been trained to a level so that in absense of trigger the policy consistently achieves high score against the oppenent while in presence of trigger it takes 'no move' action eventually achieving a very low score on average.
