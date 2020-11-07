# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:12:30 2020

@author: rasheed el-bouri
"""
from copy import copy
import numpy as np
import math
import gym
import gym_minigrid
from utils.visualiser import visualiseSteps
import argparse
from utils.boolParse import str2bool

class Node():
    """A class that creates nodes in the tree search as objects.
    Traversal of the tree can occur by following from parent to children nodes.
    After the search is concluded, the best move at each generation of the tree will by the node with the highest value."""
    
    def __init__(self, parent):
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

class MCTS():
    """ A class that creates and rollsout a monte carlo tree search.
    An initialised gym environment needs to be fed in as one of the arguments.
    Setting plot = True will display rollouts as they occur.
    Max rollout controls how many random moves are allowed during the random rollout.
    Budget determines how long the monte carlo simulation can run for."""
    
    def __init__(self, env, max_rollout=100, budget=1000):
        self.env = env # must be a gym environment
        self.copycat = copy(self.env) #copying the current environment to allow repetition with different moves
        self.root = Node(None) # initialising the root node with no parents
        self.max_rollout = max_rollout 
        self.budget = budget
        
        
    def expand(self, node):
        if (not node.children):
            for i in range(self.copycat.action_space.n):
                node.children.append(Node(node))
    
    def findLeafNodes(self):
        leaves = []
        pas = []
        node = self.root
        is_pas = True
        while is_pas:
            for child in node.children:
                if not child.children:
                    leaves.append(child)
                else:
                    pas.append(child)
            if len(pas) == 0:
                is_pas = False
            else:
                node = pas[0]
                pas.pop(0)
        
        return(leaves)
    
    def calculateUCB(self, node, epsilon):
        if node.visits == 0:
            node.visits += 1e-5
        return(node.value / node.visits + epsilon*np.sqrt(np.log(node.parent.visits)/node.visits))
    
    def calculateBestLeaf(self, epsilon):
        self.copycat = copy(self.env)
        node = self.root
        is_kids = True
        while is_kids:
            scores = []
            for kid in node.children:
                scores.append(self.calculateUCB(kid, epsilon))
            scores = [1e10 if math.isnan(x) else x for x in scores]
            best_ucb = scores.index(max(scores))
            best_leaf = node.children[best_ucb]
            self.copycat.step(best_ucb)
            if not best_leaf.children:
                is_kids = False
                return(best_leaf, scores)
            else:
                node = best_leaf
                
    
    def rolloutAndBackprop(self, node):
        sum_reward = 0
        actions=[]
        terminal=False

        while not terminal:
            action = self.copycat.action_space.sample()
            _, reward, terminal, _ = self.copycat.step(action)
            sum_reward += reward
            actions.append(action)

            if len(actions) > self.max_rollout:
                sum_reward = -1*len(actions)
                break
        
        node.visits += 1
        node.value += sum_reward

        while node.parent:
            node.parent.visits += 1
            node.parent.value += sum_reward  
            node = node.parent
        
        return(actions, sum_reward)
    
    def optimalSequence(self):
        node = self.root
        optimal_sequence = []
        while node.children:
            values = []
            for child in node.children:
                values.append(child.value)
            best_child = values.index(max(values))
            optimal_sequence.append(best_child)
            node = node.children[best_child]
        
        return(optimal_sequence)
    
        
    def runSimulation(self, exploration_rate, track_progress=False):
        for i in range(self.budget):
            if i == 0:
                self.expand(self.root)
                for child in self.root.children:
                    _,_ = self.rolloutAndBackprop(child)
            else:
                best_leaf, scores = self.calculateBestLeaf(exploration_rate)
                if best_leaf.visits == 0:
                    self.rolloutAndBackprop(best_leaf)
                else:
                    self.expand(best_leaf)
                    selection = np.random.randint(len(best_leaf.children))
                    self.rolloutAndBackprop(best_leaf.children[selection])
            if (i % 100 == 0) and track_progress:
                print("budget " + str(i) + " consumed")
        optimal_sequence = self.optimalSequence()
        return(optimal_sequence)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=int, help='How many steps allowed for rollout')
    parser.add_argument("--budget", type=int, help="How long to run the simulation")
    parser.add_argument("--exploration", type=float, help="set an exploration rate")
    parser.add_argument("--plot", type = str2bool, help="visualise rollouts")
    parser.add_argument("--verbose", type = str2bool, help="plot optimal sequence after")
    args = parser.parse_args()
    
    rollout = args.rollout
    budget = args.budget
    exploration = args.exploration
    plot = args.plot
    verbose = args.verbose
    
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env.reset()
    mct = MCTS(env, max_rollout=rollout, budget=budget)
    optimal_sequence = mct.runSimulation(exploration, track_progress=verbose)
    if plot:
        visualiseSteps(optimal_sequence)


    