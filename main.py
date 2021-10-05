import numpy
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import pygame
import math
from PIL import Image  # for creating visual of our env
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import pickle  # to save/load Q-Tables
from matplotlib import style  # to make pretty charts because it matters.
import time
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Convolution2D
# from tensorflow.keras.optimizers import Adam
# from rl.agents import DQNAgent
# from rl.memory import SequentialMemory
# from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy, BoltzmannQPolicy
# from rl.callbacks import FileLogger, ModelIntervalCheckpoint


epi=1
G=1
M_earth=200
M_rocket=1
Width=1000
Height=1000
Sun_pos = np.array([500,500])
dt=1
rocket_size = 100
earth_size = 132
GAME_SPEED =3000
def rot_center(image, angle):
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image

class OrbiterEnv(Env):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Width, Height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)
        # pygame
        self.surface = pygame.image.load("rocket.png")
        self.surface = pygame.transform.scale(self.surface, (rocket_size, rocket_size))
        self.rotate_surface = self.surface
        # Actions we can take, down, stay, up
        self.action_space = Discrete(3)
        # Temperature array
        self.observation_space = Box(np.array([0,0]), np.array([11,60]))


        self.pos = numpy.array([700.,500.])
        self.vel = numpy.array([0.,-1.])
        self.acc = numpy.array([0.,0.])
        self.goal = 300 + 10 * random.randint(5,10)
        self.transfer_period = 5000
        self.center = [self.pos[0] - rocket_size/2, self.pos[1] - rocket_size/2]
        self.earth = pygame.image.load("earth.png")
        self.earth = pygame.transform.scale(self.earth, (120, 120))
        self.alt_earth = pygame.image.load("earth-alt.jpg")
        self.alt_earth = pygame.transform.scale(self.alt_earth, (earth_size, earth_size))
        self.game_speed = 3000
        self.angle = 0
        self.rel_angle=0
        self.epi=1
        self.display_action=0
        self.angle_bt=90
        self.in_orbit_count=0
    def aepi(self):
        self.epi +=1

    def draw(self, screen):
        screen.blit(self.rotate_surface, self.center)
    def draw_text(self, screen):
        text1 = self.font.render("Episode: " + str(self.epi), True, (0, 0, 0))
        text_rect1 = text1.get_rect()
        text_rect1.center = (100, 10)
        self.screen.blit(text1, text_rect1)
        text2 = self.font.render("Action: " + str(self.display_action), True, (0, 0, 0))
        text_rect2 = text2.get_rect()
        text_rect2.center = (100, 50)
        self.screen.blit(text2, text_rect2)
        text3 = self.font.render("Angle: " + str(abs(90-self.angle_bt)), True, (0, 0, 0))
        text_rect3 = text3.get_rect()
        text_rect3.center = (100, 90)
        self.screen.blit(text3, text_rect3)
        text4 = self.font.render("Target-Vel " + str(round((np.linalg.norm(self.vel)-(G*M_earth/self.goal)**0.5), 3)), True, (0, 0, 0))
        text_rect4 = text4.get_rect()
        text_rect4.center = (100, 130)
        self.screen.blit(text4, text_rect4)
    def get_direction(self):
        r=np.subtract(Sun_pos,self.pos)
        return r/np.linalg.norm(r)

    def get_rkt_direction(self):
        return self.vel/np.linalg.norm(self.vel)

    def get_distance(self):
        return np.linalg.norm(np.subtract(Sun_pos,self.pos))

    def get_grav_acc(self):
        return G*M_earth/self.get_distance()/self.get_distance()*self.get_direction()

    def in_orbit(self):
        vel=(G*M_earth/self.goal)**0.5
        return abs(90-self.angle_bt)<3 and abs(np.linalg.norm(self.vel)-vel)<0.05*vel

    def update(self):
        self.rotate_surface = rot_center(self.surface, self.angle+7)
        self.angle_bt = int(math.degrees(np.arccos(np.dot(self.get_direction(),self.get_rkt_direction()))))

        if self.vel[0] < 0:
            self.angle = 90-math.degrees(np.arctan(self.vel[1]/self.vel[0]))
        else:
            self.angle = 270 -math.degrees(np.arctan(self.vel[1]/self.vel[0]))
        # if self.pos[0] < 500:
        #     self.rel_angle = 90 - math.degrees(np.arctan((self.pos[1]-500)/(self.pos[0]-500)))
        # else:
        #     self.rel_angle = 270- math.degrees(np.arctan((self.pos[1] - 500) / (self.pos[0] - 500)))
        self.vel += self.acc * dt
        self.pos += self.vel * dt
        self.center = [self.pos[0] - rocket_size/2, self.pos[1] - rocket_size/2]
        self.transfer_period -= 1

    def evaluate(self):
        score = 1 - abs(self.goal - self.get_distance()) / self.goal
        reward=0
        if(self.get_distance()<200):
            reward= -1
        elif(self.get_distance()<=self.goal):
            reward= 0.03*(100**score)
        else:
            reward= -0.03*(100**(1-score))


        if (self.display_action == 0):
            reward -= 0.5
        done=False

        if self.transfer_period <= 0:
            done=True
        if self.get_distance()<50:
            reward =-12000
            done=True
        # elif self.get_distance()>self.goal+200:
        #     reward =-12000
            done=True
        elif self.angle_bt < 20:
            reward = -12000
            done=True
        elif int(score*100) == 100 and self.in_orbit() :
            reward = 12000
            self.in_orbit_count +=1
            if self.in_orbit_count >= 12:
                done=True

        return reward,done

    def step(self,action):
        kscore = int((1 - abs((self.goal - self.get_distance() ))/ self.goal)*10)
        # self.acc = self.get_grav_acc()
        self.acc = np.add((action - 1) * 0.002 * self.get_rkt_direction(), self.get_grav_acc())
        self.display_action=action

        self.update()
        reward,done=self.evaluate()

        return tuple([kscore,int(self.angle_bt/3)]),reward,done,{}#,int(np.linalg.norm(self.vel))*10])

    def render(self, mode='human', close=False):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        self.screen.fill((255, 255, 255))
        if self.get_distance()>190:
            self.screen.blit(self.earth, (440, 440))
        else:
            self.screen.blit(self.alt_earth, (500-earth_size/2, 500-earth_size/2))

        pygame.draw.circle(self.screen, (0, 255, 0), Sun_pos, 200, 2)
        pygame.draw.circle(self.screen, (0, 0, 255), Sun_pos, self.goal, 2)
        self.draw(self.screen)
        self.draw_text(self.screen)



        pygame.display.flip()
        self.clock.tick(self.game_speed)

    def reset(self):
        self.pos = numpy.array([700., 500.])
        self.vel = numpy.array([0., -1.])
        self.acc = numpy.array([0., 0.])
        self.goal = 300 + 10 * random.randint(5, 10)
        self.transfer_period = 5000
        self.angle = 0
        self.rel_angle = 0
        self.angle_bt = 90
        self.in_orbit_count =0


        return tuple([int((1 - (self.goal - self.get_distance()) / self.goal)*10),int(self.angle_bt/3)])#int(np.linalg.norm(self.vel)*10)])


def simulate():
    global epsilon, epsilon_decay
    episode_rewards = []
    for episode in range(MAX_EPISODES):

        # Init environment
        state = env.reset()
        total_reward = 0

        # AI tries up to MAX_TRY times
        for t in range(MAX_TRY):

            # In the beginning, do random action to learn
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Do action and get result
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Get correspond q value from state, action pair
            q_value = q_table[state][action]
            best_q = np.max(q_table[next_state])

            # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
            q_table[state][action] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

            # Set up for the next iteration
            state = next_state

            # Draw games
            env.render()

            # When episode is done, print reward
            if done or t >= MAX_TRY - 1:
                episode_rewards.append(total_reward)
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, total_reward))
                env.aepi()
                break

        # exploring rate decay
        if epsilon >= 0.005:
            epsilon *= epsilon_decay
    moving_avg = np.convolve(episode_rewards, np.ones((1)) / 1, mode='valid')

    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel(f"Reward {1}ma")
    plt.xlabel("episode #")
    plt.show()

    with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table, f)
# def build_model(states,actions):
#     model = Sequential()
#     model.add(Flatten())
#     model.add(Dense(32, activation='relu',input_shape=states))
#     model.add(Dense(32, activation='relu'))

#     model.add(Dense(actions, activation='linear'))

#     return model
# def build_agent(model, actions):
#     policy1 = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=10000)
#     policy2 = BoltzmannQPolicy()
#     memory = SequentialMemory(limit=50000, window_length=1)

#     dqn = DQNAgent(model=model,
#                    memory=memory,
#                    policy=policy1,
#                    target_model_update=1e-2,
#                    nb_actions=actions,
#                    nb_steps_warmup=100
#                   )
#     return dqn

# def build_callbacks(env_name):
#     checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
#     log_filename = 'dqn_{}_log.json'.format(env_name)
#     callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
#     callbacks += [FileLogger(log_filename, interval=100)]
#     return callbacks

if __name__ == "__main__":
    env = OrbiterEnv()
    # states=env.observation_space.shape
    # actions= env.action_space.n
    # model = build_model(states, actions)

    # dqn = build_agent(model, actions)
    # dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    # callbacks = build_callbacks('AOTS')
    # model.summary()
    # dqn.fit(env, nb_steps=50000,
    #         visualize=False,
    #         verbose=1,
    #         callbacks=callbacks)
    # scores = dqn.test(env, nb_episodes=10, visualize=True)
    # print(np.mean(scores.history['episode_reward']))
    # dqn.save_weights('dqn_weights.h5f')


    MAX_EPISODES = 9999
    MAX_TRY = 5000
    epsilon = 1
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    q_table = np.zeros(num_box + (env.action_space.n,))
    simulate()