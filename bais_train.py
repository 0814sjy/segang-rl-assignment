import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
#import wandb
import random

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        

        nn.init.zeros_(self.fc1.bias.data)
        nn.init.zeros_(self.fc2.bias.data)
        nn.init.zeros_(self.fc3.bias.data)

    def forward(self, state):
        a = F.tanh(self.fc1(state))
        a = F.tanh(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        nn.init.zeros_(self.fc1.bias.data)
        nn.init.zeros_(self.fc2.bias.data)
        nn.init.zeros_(self.fc3.bias.data)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q

class ReplayBuffer():
    def __init__(self, state_dim, action_dim, buffer_size):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
    
        self.state = np.zeros((buffer_size, state_dim))
        self.action = np.zeros((buffer_size, action_dim))
        self.reward = np.zeros((buffer_size, 1))
        self.next_state = np.zeros((buffer_size, state_dim))
        self.done = np.zeros((buffer_size, 1))
    
    
    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done 
    
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[idx]).to(DEVICE),
            torch.FloatTensor(self.action[idx]).to(DEVICE),
            torch.FloatTensor(self.reward[idx]).to(DEVICE),
            torch.FloatTensor(self.next_state[idx]).to(DEVICE),
            torch.FloatTensor(self.done[idx]).to(DEVICE)
        )

class TD3():
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.counter = 0

        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(DEVICE)
        
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_1_target = Critic(state_dim, action_dim).to(DEVICE)
        
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2_target = Critic(state_dim, action_dim).to(DEVICE)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=0.0001)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=0.0001)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

    def select_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        return self.actor(state).cpu().data.numpy().flatten()

    def update_critic(self, critic, state, action, target_Q, critic_optimizer):
        current_Q = critic(state, action)
        loss_Q = F.mse_loss(current_Q, target_Q)
        critic_optimizer.zero_grad()
        loss_Q.backward()
        critic_optimizer.step()
    
    def hard_replacement(self, model, target_model):
        target_model.load_state_dict(model.state_dict())

    def soft_replacement(self, model, target_model, tau):
        for param, target_param in zip(model.parameters(), target_model.parameters()):
                target_param.data.copy_(((1 - tau) * target_param.data) + tau * param.data)

    def update(self, replay_buffer, episode, batchsize=256, policy_noise = 0.2, delay_freq = 2, tau = 0.005, gamma = 0.99):
        self.counter += 1
        
        with torch.no_grad():
            state, action, reward, next_state, done = replay_buffer.sample(batchsize)
            noise = (torch.ones_like(action).data.normal_(0, policy_noise).to(DEVICE)).clamp(-0.5, 0.5)
            next_action = (
					self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)
        target_Q1 = self.critic_1_target(next_state, next_action)
        target_Q2 = self.critic_2_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * gamma * target_Q).detach()

        self.update_critic(self.critic_1, state, action, target_Q, self.critic_1_optimizer)
        self.update_critic(self.critic_2, state, action, target_Q, self.critic_2_optimizer)

        if self.counter % delay_freq == 0:
            
            self.counter = 0
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.soft_replacement(self.critic_1, self.critic_1_target, tau)
            self.soft_replacement(self.critic_2, self.critic_2_target, tau)
            self.soft_replacement(self.actor, self.actor_target, tau)


    def save(self, actor_path, critic_1_path, critic_2_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic_1.state_dict(), critic_1_path)
        torch.save(self.critic_2.state_dict(), critic_2_path)
    
    def load(self, actor_path, critic_1_path, critic_2_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic_1.load_state_dict(torch.load(critic_1_path))
        self.critic_1.load_state_dict(torch.load(critic_2_path))

def seed_all(seed):
    '''
    시드 고정을 위한 함수
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if DEVICE == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate(env_name, agent, seed, max_steps, eval_iterations):
    env = gym.make(env_name)
    seed = seed + 9999
    scores = []
    for _ in range(eval_iterations):
        steps = 0 
        s, done, ret = env.reset()[0], False, 0
        while not done:
            if steps < max_steps:
                steps += 1
                a = agent.select_action(s)
                next_state, reward, done, info, _ = env.step(a)
                ret += reward
                s = next_state
                if ret >= 300 :
                    done = True
            else :
                done = True
        scores.append(ret)
    env.close()
    return round(np.mean(scores), 4)

def train(env_name, random_seed, buffer_size, max_episode, expl_noise, policy_noise, batchsize, delay_freq, tau, gamma, max_steps, max_rewards,
         wand_project, wand_group_name, wandb_name):
    try:
        env = gym.make(env_name)
    
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
    
        replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        agent = TD3(state_dim, action_dim, max_action)
    
        seed_all(random_seed)

        print(wand_project,wand_group_name,wandb_name)
        #wandb.init(project=wand_project, group=wand_group_name, name=wandb_name)

        
        all_ep_r = []
        all_steps = 0
        max_steps = 1600
        
        for episode in range(max_episode):
        
            s, done = env.reset()[0], False
            ep_r = 0
            expl_noise = expl_noise * 0.999
            #policy_noise *= 0.999
            steps = 0
            '''Interact & trian'''

            while not done:
                if steps < max_steps: 
                    steps += 1
                    a = (agent.select_action(s) + np.random.normal(0, max_action * expl_noise, size = action_dim)).clip(-max_action, max_action)
                    s_prime, r, done, info, _ = env.step(a)
                    #replay_buffer.add(s, a, r, s_prime, done)
                    # Tricks for BipedalWalker
                    if r <= -100:
                        replay_buffer.add(s, a, -1, s_prime, done)
                    else :
                        if steps == max_steps :
                            done = True
                        replay_buffer.add(s, a, r, s_prime, done)
                    s = s_prime
                    ep_r += r
                
                if replay_buffer.size > max_steps * 2:
                    agent.update(replay_buffer, episode, batchsize, policy_noise, delay_freq, tau, gamma)

                # 매 약 30 * 1600 에피소드 마다 평균 성능 제시
                if all_steps != 0 and all_steps % max_steps * 30:
                    print("")
                    #wandb.log({'Episode': episode, 'all_steps' : all_steps, '30_mean_by_steps' : np.mean(all_ep_r[-30:])})

            all_steps += steps
            all_ep_r.append(ep_r)
            
            mean_50 = np.mean(all_ep_r[-50:])
            print('seed:', random_seed,'Episode:', episode,'all_steps:', all_steps ,'steps:', steps ,'score:', ep_r, 'mean:', mean_50)    
            #wandb.log({'Episode': episode,'all_steps' : all_steps, 'ep_r': ep_r, '50_mean_by_episode' : mean_50})

            if mean_50 > 300 :
                if evaluate(env_name, agent, random_seed, max_steps, eval_iterations = 10) > 300:
                    print("학습종료")
                    break

    except Exception as e:
        print(e)
        env.close()
        #wandb.finish()
        return agent

    return agent

DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env_name = 'BipedalWalker-v3'
    model_name = "BIAS_0"
    
    random_seeds = [1, 42, 66043]
    for random_seed in random_seeds :

        train_args = {
        'buffer_size' : 320000,
        'max_episode' : 1000,
        'expl_noise' : 0.2,
        'policy_noise' : 0.2,
        'batchsize' : 256,
        'delay_freq' : 2,
        'tau' : 0.005,
        'gamma' : 0.99,
        'max_steps' : 1600,
        'max_rewards' : 300 
        }
        wandb_args = {
            "wand_project" : "RL-assign",
            "wand_group_name" : env_name + "_" + model_name,
            "wandb_name" : model_name +"_"+ str(random_seed)
        }
        
        agent = train(env_name, random_seed, **train_args, **wandb_args)
    
        save_args = {
            'actor_path' : env_name    +"_" + model_name + "_" + "actor" + "_" + str(random_seed), 
            'critic_1_path' : env_name +"_" + model_name + "_" + "critic_1" + "_" + str(random_seed) , 
            'critic_2_path' : env_name +"_" + model_name + "_" + "critic_2" + "_"  +str(random_seed)
        }
        agent.save(**save_args)

if __name__ == "__main__":
    main()