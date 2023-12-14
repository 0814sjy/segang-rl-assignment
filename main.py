#import sys
#sys.path.append('D:\\jupyter')

from init_orth_with_bais_train import TD3
import gymnasium as gym

'''
학습된 모델 테스트
'''
env_name = 'BipedalWalker-v3'
env = gym.make(env_name, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = TD3(state_dim, action_dim, max_action)

actor_path = "a66043_actor.pth"
critic_1_path = "a66043_critic_1.pth"
critic_2_path = "a66043_critic_2.pth"

save_args = {
    'actor_path' : actor_path, 
    'critic_1_path' : critic_1_path, 
    'critic_2_path' : critic_2_path
}

agent.load(**save_args)

total_reward = 0
s, done = env.reset()[0], False
while not done :
    a = agent.select_action(s)
    s_prime, r, done, info, _ = env.step(a)
    s = s_prime
    total_reward += r
env.close()
print(total_reward)