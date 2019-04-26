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
        train = False
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
                if i == 0:
                    agent.state.p_pos = np.array([0.8, 0.8])
                if i == 1:
                    agent.state.p_pos = np.array([0.8, -0.8])
                if i == 2:
                    agent.state.p_pos = np.array([-0.8, 0.8])
                if i == 3:
                    agent.state.p_pos = np.array([-0.8, -0.8])
            
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            
        for i, landmark in enumerate(world.landmarks):
            if train:
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            else:
                if i == 2:
                    landmark.state.p_pos = np.array([0.8, 0.8])
                if i == 3:
                    landmark.state.p_pos = np.array([0.8, -0.8])
                if i == 0:
                    landmark.state.p_pos = np.array([-0.8, 0.8])
                if i == 1:
                    landmark.state.p_pos = np.array([-0.8, -0.8])
            
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

        for i in range(len(world.landmarks)):
            dist = np.sqrt(np.sum(np.square(world.agents[i].state.p_pos - world.landmarks[i].state.p_pos)))            
            if(dist < ( world.agents[i].size + world.landmarks[i].size) ):
                rew += 15
            else:
            	rew -= dist

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 15
        return rew
    
    def reward(self, agent, world, world_before):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0

        for i in range(len(world.landmarks)):
            dist_before = np.sqrt(np.sum(np.square(world_before.agents[i].state.p_pos - world_before.landmarks[i].state.p_pos)))
            dist = np.sqrt(np.sum(np.square(world.agents[i].state.p_pos - world.landmarks[i].state.p_pos)))

            if(dist < ( world.agents[i].size + world.landmarks[i].size) ):
                rew += 15
                world.agents[i].state.reach = True
            else:
        	      rew -= (dist-dist_before)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
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
                xy_pos = np.dot(R, delta)
                p = np.sqrt(np.sum(np.square(xy_pos)))
                alpha = math.atan2(xy_pos[1], xy_pos[0])
                entity_pos.append([p,alpha])

        # communication of all other agents
        comm = []
        other_pos = []
        other_theta = []
        self_theta = []
        for other in world.agents:
            if other is agent: 
                self_theta.append(np.array([agent.state.theta]))
                continue
            comm.append(other.state.c)
            other_theta.append(np.array([other.state.theta]))
            theta = agent.state.theta
            delta = other.state.p_pos - agent.state.p_pos
            R = np.array([[math.cos(theta),math.sin(theta)],[-math.sin(theta),math.cos(theta)]])
            xy_pos = np.dot(R, delta)
            p = np.sqrt(np.sum(np.square(xy_pos)))           
            alpha = math.atan2(xy_pos[1], xy_pos[0])
            other_pos.append([p,alpha])
     
        return np.concatenate(entity_pos + other_pos)
