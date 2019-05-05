import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random
import math

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_adversaries = 1
        num_landmarks = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            if i < num_adversaries:
                agent.adversary = True
            else:
                agent.adversary = False
            # agent.u_noise = 1e-1
            # agent.c_noise = 1e-1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.color[i + 1] += 0.8
            landmark.index = i
        # set goal landmark
        goal = np.random.choice(world.landmarks)
        for i, agent in enumerate(world.agents):
            agent.goal_a = goal
            agent.color = np.array([0.25, 0.25, 0.25])
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
            else:
                j = goal.index
                agent.color[j + 1] += 0.5
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world, world_before):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # the distance to the goal
        return -np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))

    def adversary_reward(self, agent, world):
        # keep the nearest good agents away from the goal
        agent_dist = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in world.agents if not a.adversary]
        pos_rew = min(agent_dist)
        #nearest_agent = world.good_agents[np.argmin(agent_dist)]
        #neg_rew = np.sqrt(np.sum(np.square(nearest_agent.state.p_pos - agent.state.p_pos)))
        neg_rew = np.sqrt(np.sum(np.square(agent.goal_a.state.p_pos - agent.state.p_pos)))
        #neg_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in world.good_agents])
        return pos_rew - neg_rew
        
    def pos_in_agentaxis(self, agent, entity):
        theta = agent.state.theta
        delta = entity.state.p_pos - agent.state.p_pos
        R = np.array([[math.cos(theta), math.sin(theta)],[-math.sin(theta), math.cos(theta)]])
        xy_pos = np.dot(R, delta)
        p = np.sqrt(np.sum(np.square(xy_pos)))
        alpha = math.atan2(xy_pos[1], xy_pos[0])
        return [p, alpha]    
                   
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(self.pos_in_agentaxis(agent, entity))
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []

        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(self.pos_in_agentaxis(agent, other))
            
        goal_pos = np.array(self.pos_in_agentaxis(agent, agent.goal_a))
        self_theta = np.array([agent.state.theta])
        
        if not agent.adversary:
            #return np.concatenate(self_theta + [agent.state.p_vel] + goal_pos + [agent.color] + entity_pos + entity_color + other_pos)
            return np.concatenate(goal_pos + entity_pos + other_pos)
        else:
            #other_pos = list(reversed(other_pos)) if random.uniform(0,1) > 0.5 else other_pos  # randomize position of other agents in adversary network
            #return np.concatenate(self_theta + [agent.state.p_vel] + entity_pos + other_pos)
            return np.concatenate(entity_pos + other_pos)
