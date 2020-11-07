# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 14:23:18 2020

@author: rashe
"""
import gym
import gym_minigrid
import matplotlib.pyplot as plt


def visualiseSteps(optimal_sequence):
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env.reset()
    for step in optimal_sequence:
    
        env.step(step)
        img = env.render("rgb_array")
        plt.imshow(img)
        plt.show()