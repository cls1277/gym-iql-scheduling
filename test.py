# Date: 2024/8/1 9:13
# Author: cls1277
# Email: cls1277@163.com

from env.MachineEnv import MachineEnv
from iql.iql_dqn import *

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
episodes = 5000
load_from_checkpoint = True
run_name = "cls_test"

if load_from_checkpoint:
    for agent in agent_dict.keys():
        agent_dict[agent].load_model(f"{run_name}_{agent}_model")
env = MachineEnv()
# # Disable random exploration
for agent in agent_dict.keys():
    agent_dict[agent].epsilon = 0
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
    episode_score_dict = {
        "VTM": 0,
        "ATM": 0,
    }
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        # Tag observations with the agent index
        if agent_dict[agent].agent_index is not None:
            observation = np.append(observation, [agent_dict[agent].agent_index])

        action = agent_dict.get(agent).policy(observation, done)
        env.step(action, agent_dict[agent].agent_index)
        episode_score_dict[agent] += reward
    for agent in episode_score_dict.keys():
        last_hundred_score_dict[agent].append(episode_score_dict[agent])
        score_history_dict[agent].append(episode_score_dict[agent])
        mean_last_hundred = sum(last_hundred_score_dict[agent]) / len(last_hundred_score_dict[agent])
        rolling_means_dict[agent].append(mean_last_hundred)
    if episode % 100 == 0:
        print("Mean scores last 100 episodes:")
        for agent in rolling_means_dict.keys():
            print(f"{agent}: {rolling_means_dict[agent][-1]}")
print(f"Overall mean scores:")
for agent in score_history_dict:
    mean = sum(score_history_dict[agent]) / len(score_history_dict[agent])
    print(f"{agent}: {mean}")