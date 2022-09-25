import numpy as np

from tool.dqn import DQN
from tool.env import *
import datetime
from tqdm import tqdm
from tool.utils import set_seed

if __name__ == '__main__':
    config = dict()
    config['device'] = 'cuda'
    config['layer_sizes'] = (5, 32, 64, 32, 5)
    config['memory_size'] = 20000
    config['batch_size'] = 1024
    config['epsilon'] = 0.5
    config['epsilon_decay_coefficient'] = 0.5
    config['epsilon_decay_interval'] = 100
    config['gamma'] = 0.9
    config['lr'] = 1.e-2
    config['l2'] = 1.e-4
    config['target_replace_interval'] = 100
    dqn = DQN(config)
    n_episode = 1000
    reward_avg = []
    history = []
    best_reward = 0
    for i in range(n_episode):
        dqn.add_episode()
        js = JobShop()
        job, equ, s = js.reset()
        reward = 0
        while True:
            a = dqn.choose_action(s)
            r, s_, job, equ, done = js.step(job, equ, a)
            dqn.store_transition(s, a, r, s_, done)
            reward += r
            s = s_
            if done:
                reward_avg.append(reward)
                history.append((i, reward, np.mean(reward_avg[-100:])))
                print('Episode{:d}---Reward: {:.4f}---Average 100 reward: {:.4f}'.
                      format(i, reward, np.mean(reward_avg[-100:])))
                break
        if dqn.memory_counter > dqn.batch_size:
            dqn.learn()
        if reward > best_reward:
            dqn.save('out/env_and_agent.pt')
            best_reward = reward

    history = np.array(history, dtype=np.float32)
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(11, 4))
    ax.plot(history[:, 0], history[:, 1], label='Reward')
    ax.plot(history[:, 0], history[:, 2], label='Reward(Avg 100)')
    ax.set_xlabel('Episode', fontsize=17)
    ax.set_ylabel('Sum of reward', fontsize=17)
    ax.legend(fontsize=17)
    plt.savefig('out/reward.png', dpi=300)
    plt.close(fig)
    pd.DataFrame(history, columns=["epoch", "reward", "average_100_reward"]).to_csv(
        f'out/loss.csv', index=False)
