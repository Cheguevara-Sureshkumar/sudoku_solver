#!usr/bin/env python3

import numpy as np 

class SudokuSmallAgent:
    def __init__(self, alpha=0.5, gamma=0.7, epsilon=0.5):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Use a dictionary for sparse storage: {state_index: array_of_64_q_values}
        self.q_table = {}

    def encode_state(self, board):
        # Convert board to a string or tuple to use as a dictionary key
        # Using a tuple is efficient and avoids floating point index issues
        return tuple(np.array(board).flatten())
    
    def get_q_values(self, state):
        if state not in self.q_table:
            # Initialize with zeros for 64 actions (16 cells * 4 values)
            self.q_table[state] = np.zeros(64)
        return self.q_table[state]

    def select_action(self, state, valid_actions_dict):
        """
        state: encoded state (tuple)
        valid_actions_dict: {cell_index: [possible_values]} from env
        """
        q_values = self.get_q_values(state)
        
        # Exploration vs Exploitation
        if np.random.rand() < self.epsilon:
            # print(list(valid_actions_dict.keys()))
            # Random valid action
            cell = np.random.choice(list(valid_actions_dict.keys()))
            val = np.random.choice(valid_actions_dict[cell])
        else:
            # Best valid action
            best_q = -float('inf')
            cell, val = None, None
            
            for c, vals in valid_actions_dict.items():
                for v in vals:
                    # Map (cell, value) to index 0-63
                    # cell is 0-15, value is 1-4. Index = cell*4 + (value-1)
                    idx = c * 4 + (v - 1)
                    if q_values[idx] > best_q:
                        best_q = q_values[idx]
                        cell, val = c, v
            
            # If no better action found, pick random valid
            if cell is None:
                # print(list(valid_actions_dict.keys()))
                cell = np.random.choice(list(valid_actions_dict.keys()))
                val = np.random.choice(valid_actions_dict[cell])

        return cell, val

    def update_q_value(self, state, action_cell, action_val, reward, next_state, valid_actions_dict):
        current_q = self.q_table[state][action_cell * 4 + (action_val - 1)]
        if next_state is None:
            max_next_q = 0
        else:
            next_q_values = self.get_q_values(next_state)

            max_next_q = -float('inf')

            for c, vals in valid_actions_dict.items():
                for v in vals:
                    idx = c * 4 + (v - 1)
                    if next_q_values[idx] > max_next_q:
                        max_next_q = next_q_values[idx]

            # If no valid actions were found, fall back to 0
            if max_next_q == -float('inf'):
                max_next_q = 0

        target = reward + self.gamma * max_next_q
        self.td_error = abs(target - current_q)
        
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[state][action_cell * 4 + (action_val - 1)] = new_q
    
    def save_table(self, filename="q_table.npy"):
        np.save(filename, self.q_table)

    def load_table(self, filename="q_table.npy"):
        self.q_table = np.load(filename, allow_pickle=True).item()

    def decay_learning_rate(self):
        self.alpha = max(self.alpha * 0.99, 0.1)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * 0.99, 0.1)
