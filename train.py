# Date: 2024/7/31 15:21
# Author: cls1277
# Email: cls1277@163.com

# from pettingzoo.mpe import simple_adversary_v3
import time
import matplotlib.pyplot as plt
from iql.iql_dqn import *
import os
from env.MachineEnv import MachineEnv
os.environ['KMP_DUPLICATE_LIB_OK']='True'

VTM = DQNAgent(epsilon=1.0, min_epsilon=0.1, decay_rate=0.999,
                   learning_rate=0.0001, gamma=0.99, batch_size=64,
                   tau=0.001, q_network=VTMAgentDQN(), target_network=VTMAgentDQN(),
                   max_memory_length=100000, agent_index=0)

ATM = DQNAgent(epsilon=1.0, min_epsilon=0.1, decay_rate=0.999,
                       learning_rate=0.0001, gamma=0.99, batch_size=64,
                       tau=0.001, q_network=ATMAgentDQN(), target_network=ATMAgentDQN(),
                       max_memory_length=100000, agent_index=1)

dqn_agent_dict = {
    "VTM": VTM,
    "ATM": ATM,
}

agent_dict = dqn_agent_dict
episodes = 100
prioritized = True
run_name = "cls_test"

env = MachineEnv()
# env = simple_adversary_v3.env(N=2, max_cycles=50)
for agent in agent_dict.keys():
    agent_dict[agent].prioritized_memory.beta_annealing_steps = 50*episodes
start = time.time()
last_hundred_score_dict = {
    "VTM": deque(maxlen=100),
    "ATM": deque(maxlen=100),
}
score_history_dict = {
    "VTM": [],
    "ATM": [],
}
rolling_means_dict = {
    "VTM": [],
    "ATM": [],
}
for episode in range(episodes):
    env.reset()
    cycles = 0
    episode_score_dict = {
        "VTM": 0,
        "ATM": 0,
    }
    for agent in env.agent_iter():
        agent_obj = agent_dict.get(agent)
        observation, reward, done, info = env.last()
        # Tag observations with the agent index
        if agent_obj.agent_index is not None:
            observation = np.append(observation, [agent_obj.agent_index])

        action = agent_obj.policy(observation, done)
        env.step(action, agent_obj.agent_index)

        if agent_obj.last_observation is not None and agent_obj.last_action is not None:
            agent_obj.push_memory(TransitionMemory(agent_obj.last_observation, agent_obj.last_action, reward, observation, done))
            agent_obj.prioritized_memory.push(TransitionMemory(agent_obj.last_observation, agent_obj.last_action, reward, observation, done))

        if prioritized:
            agent_obj.do_prioritized_training_update((episode * 50) + cycles)
        else:
            agent_obj.do_training_update()

        agent_obj.last_observation = observation
        agent_obj.last_action = action

        if cycles % 5 == 0:
            agent_obj.update_target_network()

        agent_obj.decay_epsilon()
        episode_score_dict[agent] += reward

        cycles += 1
    for agent in episode_score_dict.keys():
        last_hundred_score_dict[agent].append(episode_score_dict[agent])
        score_history_dict[agent].append(episode_score_dict[agent])
        mean_last_hundred = sum(last_hundred_score_dict[agent])/len(last_hundred_score_dict[agent])
        rolling_means_dict[agent].append(mean_last_hundred)
    if episode % 100 == 0:
        print(f"Episode {episode} complete!")
        for agent in agent_dict.keys():
            agent_dict[agent].save_model(f"{run_name}_{agent}_model")
        print(f"PER: {prioritized}")
        print("Mean scores last 100 episodes:")
        for agent in rolling_means_dict.keys():
            print(f"{agent}: {rolling_means_dict[agent][-1]}")
        print("\n")
print(f'Time elapsed: {time.time() - start} seconds')