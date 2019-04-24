import numpy as np
import seaborn as sns
import math

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity linar velocity angular velocity
        self.p_vel = None
        # axis
        self.theta = 0

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties of wall entities
class Wall(object):
    def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1,
                 hard=True):
        # orientation: 'H'orizontal or 'V'ertical
        self.orient = orient
        # position along axis which wall lays on (y-axis for H, x-axis for V)
        self.axis_pos = axis_pos
        # endpoints of wall (x-coords for H, y-coords for V)
        self.endpoints = np.array(endpoints)
        # width of wall
        self.width = width
        # whether wall is impassable to all agents
        self.hard = hard
        # color of wall
        self.color = np.array([0.0, 0.0, 0.0])


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # color
        self.color = None
        # max speed
        self.max_linear_speed = 1
        # min radius
        self.max_angular_speed = 1
        # accel
        self.accel = None
        # state: including internal/mental state p_pos, p_vel
        self.state = EntityState()
        #commu channel
        self.channel = None

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agent are adversary
        self.adversary = False
        # agent are dummy
        self.dummy = False
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state: including communication state(communication utterance) c and internal/mental state p_pos, p_vel
        self.state = AgentState()
        # action: physical action u & communication action c
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.walls = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1

        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]
    

    # 新增函数 计算world中所有entity（包括agent �?landmarks）的距离并判断是否相�?    def calculate_distances(self):
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                              len(self.entities),
                                              self.dim_p))
            # calculate minimum distance for a collision between all entities （size相加�?            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        # cached_dist_vect 保存了两�?entity 之间的每一维坐标差，还未计算距�?        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        # cached_dist_mag �?cached_dist_vect 中的两两距离求平方开根，得到2维距离矩�?        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)

        # cached_collisions 是一个二�?/1矩阵�?表示两个 entity 相撞
        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)

    # 新增函数
    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        dummy_colors = [(0, 0, 0)] * n_dummies
        adv_colors = [(0.75, 0.25, 0.25)] * n_adversaries #sns.color_palette("OrRd_d", n_adversaries)
        good_colors = [(0.25, 0.25, 0.75)] * n_good_agents#sns.color_palette("GnBu_d", n_good_agents)
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color
    
    # landmark color
    def assign_landmark_colors(self):
        for landmark in self.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])
    
    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # gather forces applied to entities
        p_u = [None] * len(self.entities)
        # apply agent physical controls
        p_u = self.apply_action_u(p_u)

        # integrate physical state
        self.integrate_state(p_u)

        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
        # calculate and store distances between all entities
        if self.cache_dists:
            self.calculate_distances()


    # gather agent action forces
    def apply_action_u(self, p_u):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_u[i] = agent.action.u + noise
        return p_u

    # integrate physical state
    def integrate_state(self, p_u):
        for i, entity in enumerate(self.entities):
            if not entity.movable: continue
            if (p_u[i] is not None):
                entity.state.p_vel = p_u[i]

            #speed limit
            if entity.max_linear_speed is not None:
                # linear speed
                if abs(entity.state.p_vel[0]) > abs(entity.max_linear_speed):
                    linear_speed = entity.max_linear_speed
                    math.copysign(linear_speed, entity.state.p_vel[0])   
                    entity.state.p_vel[0] = linear_speed
            if entity.max_angular_speed is not None:   
                # angular speed
                if abs(entity.state.p_vel[1]) > abs(entity.max_angular_speed):
                    angular_speed = entity.max_angular_speed
                    math.copysign(angular_speed, entity.state.p_vel[1])   
                    entity.state.p_vel[1] = angular_speed
            
            #calculate radius
            if ( abs(entity.state.p_vel[1])< 0.00001 ):
                x = entity.state.p_vel[0] * self.dt
                y = 0
                theta_temp = 0
            else:
                r = entity.state.p_vel[0] / (entity.state.p_vel[1])
                theta_temp = entity.state.p_vel[1] * self.dt
                x = r * math.sin(theta_temp)
                y = r * (1 - math.cos(theta_temp))
            
            entity.state.p_pos[0] += x * math.cos(-entity.state.theta) + y * math.sin(-entity.state.theta)
            entity.state.p_pos[1] += y * math.cos(-entity.state.theta) - x * math.sin(-entity.state.theta)
            entity.state.theta += theta_temp
            if (entity.state.theta > (2 * math.pi)):
                entity.state.theta -= 2 * math.pi
            if (entity.state.theta < 0):
                entity.state.theta += 2 * math.pi
       

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

