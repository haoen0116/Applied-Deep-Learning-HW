# ADL Homework 3
This is the homework about the course of the Applied Deep Learning in National Taiwan University.

The task about homework 3 is to use the Reinforcement Learning (RL) model to play the game.
In this homework, we have to use the game environment in GYM to train our model, 
and let model can play the game by itself. And the games are [LunarLander] and [AssaultNoFrameskip].

##How to run :
Training policy gradient in [LunarLander] game:

    python3.6 main.py --train_pg

Testing policy gradient in [LunarLander] game:

    python3.6 test.py --test_pg

Training policy gradient improvement version in [LunarLander] game:

    python3.6 main.py --train_pg_improve

Testing policy gradient improvement version in [LunarLander] game:

    python3.6 test.py --test_pg_improve

Training DQN in [AssaultNoFrameskip] game:

    python3.6 main.py --train_dqn

Testing DQN in [AssaultNoFrameskip] game:

    python3.6 test.py --test_dqn

If you want to see your agent playing the game,

    python3.6 test.py --test_[pg|dqn] --do_render


Run improve improve_ddqn:

    cd improve_ddqn
    python3.6 improve_ddqn.py
or

    python3.6 improve_dqn.py
