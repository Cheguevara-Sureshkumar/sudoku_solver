#!usr/bin/env python3

from env import SudokuSmallEnv
from agent import SudokuSmallAgent
from puzzle_generator import SudokuSmallPuzzleGenerator
import matplotlib.pyplot as plt
import numpy as np

def train(episodes=1000):
    generator = SudokuSmallPuzzleGenerator(removals=6)
    agent = SudokuSmallAgent(0.6, 0.9, 0.9)
    try:
        agent.load_table()
    except FileNotFoundError:
        print("No existing Q-table found. Starting from scratch.")

    wins =0
    losses =0
    count =0
    td_error_per_episode = []
    

    for episode in range(episodes):
        # board = generator.puzzle
        board = np.array(generator.puzzle).copy()
        env = SudokuSmallEnv(board)
        # state = agent.encode_state(board)
        # valid_actions = env.get_valid_actions()
        # print(valid_actions)
        win = None
        total_reward_per_episode = 0
        episode_td_error = 0
        step_count = 0
        count = 0
        while win is None:
            state = agent.encode_state(board)
            valid_actions = env.get_valid_actions()
            # if(len(valid_actions) ==0 ):
            #     # print("No valid actions found")
            #     # print(board)
            #     # print(env.check_win())
            #     win = False
            #     break
            action_cell, action_val = agent.select_action(state, valid_actions)
            # new_board, reward, win = env.step(action_cell, action_val)
            new_board, reward, win = env.step(action_val, action_cell)
            total_reward_per_episode+=reward
            new_state = agent.encode_state(new_board)

            next_valid_actions = env.get_valid_actions()
            
            # print(valid_actions)
            agent.update_q_value(state, action_cell, action_val, reward, new_state, next_valid_actions)
            state = new_state
            board = new_board
            step_count += 1
            episode_td_error += abs(agent.td_error)
            # agent.decay_epsilon()
            
            # agent.decay_epsilon()
            # agent.decay_learning_rate()
            if(env.check_win()):
                break
            # if win is not None:
            #     break
        if win == True:
            wins += 1
        else:
            losses += 1
        avg_td_error = episode_td_error / step_count
        td_error_per_episode.append(avg_td_error)
        # print(count)
        if(episode % 100 == 0):
            agent.decay_epsilon()
        # agent.decay_learning_rate()
        print(f"Alpha : {agent.alpha} | Gamma : {agent.gamma} | Epsilon : {agent.epsilon:.4f} | Reward : {total_reward_per_episode} | TD Error : {avg_td_error:.4f} | Completed {episode+1}/{episodes} episodes. Win : {True if win else False}")
    
    agent.save_table()
    print(f"Number of wins : {wins}")
    print(f"Number of losses : {losses}")
    print(f"Win Percentage: {(wins/episodes)*100:.2f}%")

    # Plot TD error per episode (averaged every N episodes)
    N = 50
    td_arr = np.array(td_error_per_episode)
    # Average over chunks of N episodes
    trimmed = td_arr[:len(td_arr) // N * N]
    avg_chunks = trimmed.reshape(-1, N).mean(axis=1)
    x_vals = np.arange(1, len(avg_chunks) + 1) * N

    plt.figure(figsize=(12, 5))
    plt.plot(x_vals, avg_chunks, marker='o', markersize=3, label=f'Avg TD Error (per {N} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average TD Error')
    plt.title(f'TD Error (averaged every {N} episodes)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('td_error_plot.png')
    plt.show()

if __name__ == "__main__":
    train(8000)
    
    