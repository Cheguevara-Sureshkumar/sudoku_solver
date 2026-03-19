#!usr/bin/env python3

"""
This environment has 4×4 grid table with 2×2 sub grids, with numbers from 1 to 4.
"""

SQRT_N = 2
N = 4

import numpy as np

class SudokuSmallEnv:
    def __init__(self, puzzle):
        # Convert to numpy array for easier slicing and manipulation
        self.board = np.array(puzzle)
        # print("Board initialized:\n", self.board)

    def check_win(self):
        # 1. Check if board is full (no zeros)
        if np.any(self.board == 0):
            # 5. Check for Valid Actions
            valid_actions = self.get_valid_actions()
            if len(valid_actions) == 0:
                # print("NO Valid Actions")
                return False
            return None

        # 2. Check Rows
        for r in range(4):
            if len(np.unique(self.board[r, :])) != 4:
                return False

        # 3. Check Columns
        for c in range(4):
            if len(np.unique(self.board[:, c])) != 4:
                return False

        # 4. Check 2x2 Boxes
        for r in range(0, 4, 2):
            for c in range(0, 4, 2):
                box = self.board[r:r+2, c:c+2]
                if len(np.unique(box)) != 4:
                    return False
        
        # 5. Check for Valid Actions
        # valid_actions = self.get_valid_actions()
        # if len(valid_actions) == 0:
        #     print("NO Valid Actions")
        #     return False

        return True
    
    def get_cell(self, i, j):
        """Standard row-major indexing (0-15)"""
        return i * 4 + j
    
    def get_coord(self, n):
        """Standard row-major inverse mapping"""
        return n // 4, n % 4

    # def get_reward(self):
    #     win = self.check_win()
    #     if win is True:
    #         return 100
    #     elif win is False:
    #         return -100
    #     else: # win is None (game in progress)
    #         return -1
        # else:
        #     return -300

    def check_horizontal(self, action, cell):
        i, j = self.get_coord(cell)
        for c in range(4):
            if self.board[i, c] == 0:
                return False
        return True
    
    def check_vertical(self, action, cell):
        i, j = self.get_coord(cell)
        for r in range(4):
            if self.board[r, j] == 0:
                return False
        return True
    
    def check_box(self, action, cell):
        i, j = self.get_coord(cell)
        start_row, start_col = (i // 2) * 2, (j // 2) * 2
        for r in range(start_row, start_row + 2):
            for c in range(start_col, start_col + 2):
                if self.board[r, c] == 0:
                    return False
        return True
    
    def get_reward(self, action, cell):
        win = self.check_win()
        
        if win is True:
            return 100
        elif win is False:
            return -100
        
        # i, j = self.get_coord(cell)

        # Reward valid placement
        # if self.is_valid(self.board, i, j, action):
        #     reward = 5
        # else:
        #     return -10  # strong penalty for invalid move
        filled = np.count_nonzero(self.board)
        reward = filled * 2   # reward progress

        # Bonus for completing structures
        bonus = 0
        if self.check_horizontal(action, cell):
            bonus += 10
        if self.check_vertical(action, cell):
            bonus += 10
        if self.check_box(action, cell):
            bonus += 10

        return reward + bonus

    # def get_valid_actions(self):
    #     valid_actions = {}
    #     for i in range(4):
    #         for j in range(4):
    #             if self.board[i][j] == 0:
    #                 valid_actions[self.get_cell(i, j)] = [1, 2, 3, 4]
    #     return valid_actions
    
    def is_valid(self, board, row, col, num):
        # Row check
        if num in board[row, :]:
            return False
        # Column check
        if num in board[:, col]:
            return False
        # 2x2 Box check
        start_row, start_col = (row // 2) * 2, (col // 2) * 2
        if num in board[start_row:start_row+2, start_col:start_col+2]:
            return False

        return True

    def get_valid_actions(self):
        # print(self.board)
        valid_actions = {}
        for i in range(4):
            for j in range(4):
                if self.board[i, j] == 0:
                    possible_nums = []
                    for num in range(1, 5):
                        if self.is_valid(self.board, i, j, num):
                            possible_nums.append(num)
                    valid_actions[self.get_cell(i, j)] = possible_nums

        valid_actions = {k: v for k, v in valid_actions.items() if v}
        return valid_actions

    def step(self, action, cell):
        i, j = self.get_coord(cell)
        self.board[i, j] = action
        win = self.check_win()
        reward = self.get_reward(action, cell)
        return self.board, reward, win
            

