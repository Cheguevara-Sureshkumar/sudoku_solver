import random
import copy

N = 4
SQRT_N = 2

class SudokuSmallPuzzleGenerator:
    def __init__(self, removals=6):
        self.board = self.generate_full_board()
        self.puzzle = self.make_puzzle(self.board, removals)
        self.print_board(self.puzzle)

    def generate_full_board(self):
        board = [[0]*N for _ in range(N)]
        self.solve(board)
        return board

    def solve(self, board):
        for row in range(N):
            for col in range(N):
                if board[row][col] == 0:
                    nums = list(range(1, N+1))
                    random.shuffle(nums)
                    for num in nums:
                        if self.is_valid(board, row, col, num):
                            board[row][col] = num
                            if self.solve(board):
                                return True
                            board[row][col] = 0
                    return False
        return True

    def is_valid(self, board, row, col, num):
        # Row & column check
        for i in range(N):
            if board[row][i] == num or board[i][col] == num:
                return False

            # 2x2 box check 
            start_row = row - row % SQRT_N
            start_col = col - col % SQRT_N

        for i in range(SQRT_N):
            for j in range(SQRT_N):
                if board[start_row + i][start_col + j] == num:
                    return False

        return True


    # Count number of solutions
    def count_solutions(self, board):
        count = [0]

        def backtrack():
            if count[0] > 1:
                return
            for r in range(N):
                for c in range(N):
                    if board[r][c] == 0:
                        for num in range(1, N+1):
                            if self.is_valid(board, r, c, num):
                                board[r][c] = num
                                backtrack()
                                board[r][c] = 0
                        return
            count[0] += 1

        backtrack()
        return count[0]




    # Remove numbers while keeping uniqueness
    def make_puzzle(self,board, removals=6):
        puzzle = copy.deepcopy(board)
        attempts = removals

        while attempts > 0:
            row = random.randint(0, N-1)
            col = random.randint(0, N-1)

            if puzzle[row][col] == 0:
                continue

            backup = puzzle[row][col]
            puzzle[row][col] = 0

            # Check uniqueness
            board_copy = copy.deepcopy(puzzle)
            if self.count_solutions(board_copy) != 1:
                puzzle[row][col] = backup
                attempts -= 1

        return puzzle


    # Print board nicely
    def print_board(self, board):
        for i in range(N):
            if i % SQRT_N == 0 and i != 0:
                print("-" * 10)
            for j in range(N):
                if j % SQRT_N == 0 and j != 0:
                    print("|", end=" ")
                print(board[i][j] if board[i][j] != 0 else "_", end=" ")
            print()


# MAIN
if __name__ == "__main__":
    print("Generated Sudoku Puzzle:\n")
    generator = SudokuSmallPuzzleGenerator(removals=6)
    puzzle = generator.puzzle