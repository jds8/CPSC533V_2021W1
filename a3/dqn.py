import gym
import math
import random
from itertools import count
import torch
from eval_policy import eval_policy, device
from model import MyModel
from replay_buffer import ReplayBuffer


BATCH_SIZE = 256
GAMMA = 0.99
EPS_EXPLORATION = 0.2
TARGET_UPDATE = 10
NUM_EPISODES = 4000
TEST_INTERVAL = 25
LEARNING_RATE = 10e-4
RENDER_INTERVAL = 20
ENV_NAME = 'CartPole-v0'
PRINT_INTERVAL = 10

env = gym.make(ENV_NAME)
state_shape = len(env.reset())
n_actions = env.action_space.n

model = MyModel(state_shape, n_actions).to(device)
target = MyModel(state_shape, n_actions).to(device)
target.load_state_dict(model.state_dict())
target.eval()

l2loss_function = torch.nn.MSELoss()
# l2loss_function = torch.nn.SmoothL1Loss()
# l2loss_function = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer()

def choose_action(state, test_mode=False):
    if random.random() < EPS_EXPLORATION:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            state = torch.tensor([state], device=device, dtype=torch.double)
            return model(state).max(1)[1].view(1, 1).to(torch.long)

# def optimize_model(state, action, next_state, reward, done):
#     with torch.no_grad():
#         next_state = torch.tensor([next_state], device=device, dtype=torch.double)
#         target_q = target(next_state).max(1)[0].type(torch.double)

#     use_next = torch.tensor(not done, device=device, dtype=torch.bool)
#     next_state_action_value = target_q * use_next
#     y = (torch.tensor(reward) + GAMMA * next_state_action_value).type(torch.double)

#     estimate = model(state)[action]
#     loss = l2loss_function(estimate, y)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

def optimize_model(state, action, next_state, reward, done):

    memory.push(state, action, next_state, reward, done)

    states_batch, actions_batch, next_states, rewards_batch, dones = memory.sample(min(len(memory), BATCH_SIZE))

    # states_batch = torch.cat(states)
    # actions_batch = torch.cat(actions)
    # rewards_batch = torch.cat(rewards)

    # not_dones = torch.tensor(tuple(map(lambda s: s is not None,
    #                                       batch.next_state)), device=device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_state
    #                                             if s is not None])
    non_final_next_states = torch.cat([s for s in next_states if s is not None])
    if len(dones) == 1:
        if not dones[0]:
            not_dones = torch.tensor(0)
        else:
            not_dones = None
    else:
        not_dones = torch.cat([torch.tensor(idx, device=device, dtype=torch.bool) for idx, dn in enumerate(dones) if dn])

    with torch.no_grad():
        # next_state = torch.tensor([next_states], device=device, dtype=torch.double)
        # target_q = target(next_state).max(1)[0].type(torch.double)

        next_state_action_values = torch.zeros(BATCH_SIZE, device=device)
        if not_dones is not None:
            import pdb; pdb.set_trace()
            next_state_action_values[not_dones] = target(non_final_next_states).max(1)[0]
        else:
            next_state_action_values = target(non_final_next_states).max(1)[0]

    # next_state_action_value = target_q * use_next
    y = (torch.tensor(rewards_batch) + GAMMA * next_state_action_values).type(torch.double)

    estimate = model(states_batch).gather(1, actions_batch)
    loss = l2loss_function(estimate, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_reinforcement_learning(render=False):
    steps_done = 0
    best_score = -float("inf")

    for i_episode in range(1, NUM_EPISODES+1):
        episode_total_reward = 0
        state = env.reset()
        for t in count():
            action = choose_action(state)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0][0])
            # next_state, reward, done, _ = env.step(action.cpu().numpy())
            steps_done += 1
            episode_total_reward += reward

            optimize_model(state, action, next_state, reward, done)

            state = next_state

            if render:
                env.render(mode='human')

            if done:
                if i_episode % PRINT_INTERVAL == 0:
                    print('[Episode {:4d}/{}] [Steps {:4d}] [reward {:.1f}]'
                        .format(i_episode, NUM_EPISODES, t, episode_total_reward))
                break

        if i_episode % TARGET_UPDATE == 0:
            target.load_state_dict(model.state_dict())

        if i_episode % TEST_INTERVAL == 0:
            print('-'*10)
            score = eval_policy(policy=model, env=ENV_NAME, render=render)
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), "best_model_{}.pt".format(ENV_NAME))
                print('saving model.')
            print("[TEST Episode {}] [Average Reward {}]".format(i_episode, score))
            print('-'*10)


if __name__ == "__main__":
    train_reinforcement_learning()
