# Date: 2024/8/1 9:13
# Author: cls1277
# Email: cls1277@163.com

from pettingzoo.mpe import simple_adversary_v3
from iql.iql_dqn import *

agent_0_q_net = AgentDQN()
agent_0_target_net = AgentDQN()

agent_1_q_net = AgentDQN()
agent_1_target_net = AgentDQN()

agent_0 = DQNAgent(epsilon=1.0, min_epsilon=0.1, decay_rate=0.999,
                   learning_rate=0.0001, gamma=0.99, batch_size=64,
                   tau=0.001, q_network=agent_0_q_net, target_network=agent_0_target_net,
                   max_memory_length=100000, agent_index=1)

agent_1 = DQNAgent(epsilon=1.0, min_epsilon=0.1, decay_rate=0.999,
                   learning_rate=0.0001, gamma=0.99, batch_size=64,
                   tau=0.001, q_network=agent_1_q_net, target_network=agent_1_target_net,
                   max_memory_length=100000, agent_index=2)

adversary_0 = DQNAgent(epsilon=1.0, min_epsilon=0.1, decay_rate=0.999,
                       learning_rate=0.0001, gamma=0.99, batch_size=64,
                       tau=0.001, q_network=AdversaryDQN(), target_network=AdversaryDQN(),
                       max_memory_length=100000, agent_index=3)
dqn_agent_dict = {
    "agent_0": agent_0,
    "agent_1": agent_1,
    "adversary_0": adversary_0,
}

agent_dict = dqn_agent_dict
episodes = 100
load_from_checkpoint = True
run_name = "cls_test"

if load_from_checkpoint:
    for agent in agent_dict.keys():
        agent_dict[agent].load_model(f"{run_name}_{agent}_model")
env = simple_adversary_v3.env(N=2, max_cycles=50)
# Disable random exploration
for agent in agent_dict.keys():
    agent_dict[agent].epsilon = 0
last_hundred_score_dict = {
    "agent_0": deque(maxlen=100),
    "agent_1": deque(maxlen=100),
    "adversary_0": deque(maxlen=100),
}
score_history_dict = {
    "agent_0": [],
    "agent_1": [],
    "adversary_0": [],
}
rolling_means_dict = {
    "agent_0": [],
    "agent_1": [],
    "adversary_0": [],
}
for episode in range(episodes):
    env.reset()
    episode_score_dict = {
        "agent_0": 0,
        "agent_1": 0,
        "adversary_0": 0,
    }
    for agent in env.agent_iter():
        observation, reward, _, done, info = env.last()
        # Tag observations with the agent index
        if agent_dict[agent].agent_index is not None:
            observation = np.append(observation, [agent_dict[agent].agent_index])

        action = agent_dict.get(agent).policy(observation, done)
        env.step(action)
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