"""
Taxi MAP ( 5 X 5 )
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",

Description:
There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue).
When the episode starts, the taxi starts off at a random square and the passenger is at a random location.
The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination
(another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off,
the episode ends.

Observations:
There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger
(including the case when the passenger is in the taxi), and 4 destination locations.
"""
import gym
import numpy as np

if __name__ == "__main__":
    env = gym.make('Taxi-v3')

    lr = 0.01
    discount_factor = 0.95
    num_episodes = 20000

