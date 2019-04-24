import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import math

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 4
        num_landmarks = 4
        #world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
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
        train = True
        # random properties for agents
        world.assign_agent_colors()
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        for i, agent in enumerate(world.agents):
            if train:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            else:
                if(world.dim_p == 2):
                    if i%2:
            	          agent.state.p_pos = np.array([1.0 , -2.0 + 2.0 / (len(world.agents)/2 + 1 ) * i//2])
                    else:
                        agent.state.p_pos = np.array([-1.0 , -2.0 + 2.0 / (len(world.agents)/2 + 1 ) * i//2])
                else:
            	      agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if train:
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            else:
                if(world.dim_p == 2):
            	    #landmark.state.p_pos = np.array([0.5, 0.8 - 1.6 / (len(world.landmarks) - 1 ) * i])
                   landmark.state.
                else:
            	    landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        #for l in world.landmarks:
            #dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            #rew -= min(dists)

        for i in range(len(world.landmarks)):
            dist = np.sqrt(np.sum(np.square(world.agents[i].state.p_pos - world.landmarks[i].state.p_pos)))
            #rew -= dist
            
            if(dist < ( world.agents[i].size + world.landmarks[i].size) ):
                rew += 15
                #print(rew)
            else:
            	rew -= dist

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    #rew -= 1
                    rew -= 15
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []

        for i, landmark in enumerate(world.landmarks):
            if world.agents[i] is agent:
                theta = agent.state.theta
                delta = landmark.state.p_pos - agent.state.p_pos
                R = np.array([[math.cos(theta),math.sin(theta)],[-math.sin(theta),math.cos(theta)]])
                entity_pos.append(np.dot(R,delta))

        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            theta = agent.state.theta
            delta = other.state.p_pos - agent.state.p_pos
            R = np.array([[math.cos(theta),math.sin(theta)],[-math.sin(theta),math.cos(theta)]])
            other_pos.append(np.dot(R,delta))
        return np.concatenate([agent.state.p_vel] + entity_pos + other_pos + comm)
