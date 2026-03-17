#!usr/bin/env python3

from env import SudokuSmallEnv
from agent import SudokuSmallAgent
from puzzle_generator import SudokuSmallPuzzleGenerator

def train(episodes=1000):
    generator = SudokuSmallPuzzleGenerator()
    agent = SudokuSmallAgent(0.9, 0.8, 0.95)
    try:
        agent.q_table = agent.load_table()
    except FileNotFoundError:
        print("No existing Q-table found. Starting from scratch.")

    wins =0
    losses =0
    count =0

    for episode in range(episodes):
        board = generator.puzzle
        env = SudokuSmallEnv(board)
        # state = agent.encode_state(board)
        # valid_actions = env.get_valid_actions()
        # print(valid_actions)
        win = None
        while win is None:
            state = agent.encode_state(board)
            valid_actions = env.get_valid_actions()
            if(len(valid_actions) ==0 ):
                # print("No valid actions found")
                # print(board)
                # print(env.check_win())
                win = False
                break
            action_cell, action_val = agent.select_action(state, valid_actions)
            # new_board, reward, win = env.step(action_cell, action_val)
            new_board, reward, win = env.step(action_val, action_cell)
            new_state = agent.encode_state(new_board)
            
            # print(valid_actions)
            agent.update_q_value(state, action_cell, action_val, reward, new_state, valid_actions)
            state = new_state
            board = new_board
            if(count == 500):
                agent.decay_epsilon()
                agent.decay_learning_rate()
                count = 0
            count += 1
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
        
        print(f"Completed {episode+1}/{episodes} episodes. Win : {True if win else False}")
    
    agent.save_table()
    print(f"Number of wins : {wins}")
    print(f"Number of losses : {losses}")

if __name__ == "__main__":
    train(15000)
    
    