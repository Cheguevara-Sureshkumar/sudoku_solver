from puzzle_generator import SudokuSmallPuzzleGenerator
from env import SudokuSmallEnv
from agent import SudokuSmallAgent

if __name__ == "__main__":
    
    generator = SudokuSmallPuzzleGenerator()
    puzzle = generator.puzzle
    env = SudokuSmallEnv(puzzle)
    agent = SudokuSmallAgent(0.75, 0.6, 0.0)
    agent.load_table("model/q_82.npy")
    
    win = None
    while win is None:
        state = agent.encode_state(puzzle)
        valid_actions = env.get_valid_actions()
        if(len(valid_actions) == 0):
            break
        cell, val = agent.select_action(state, valid_actions)
        new_puzzle, reward, win = env.step(val, cell)
        puzzle = new_puzzle
    print(puzzle)
    print(env.check_win())
