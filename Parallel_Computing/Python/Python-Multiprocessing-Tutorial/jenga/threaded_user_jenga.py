import random
import time
import threading
from datetime import datetime

class Jenga:
    def __init__(self, total_blocks=5000):
        self.total_blocks = total_blocks
        self.blocks = [i for i in range(1, total_blocks + 1)]
        self.remaining_blocks = total_blocks

    def reset_game(self):
        self.blocks = [i for i in range(1, self.total_blocks + 1)]
        self.remaining_blocks = self.total_blocks

    def remove_block(self):
        if self.remaining_blocks <= 0:
            return False

        # Randomly pick a block to remove
        block_to_remove = random.choice(self.blocks)
        self.blocks.remove(block_to_remove)
        self.remaining_blocks -= 1

        # Random chance for the tower to fall
        if random.random() > (self.remaining_blocks / self.total_blocks):
            return False

        return True

    def play_game(self):
        while self.remaining_blocks > 0:
            if not self.remove_block():
                break


def run_jenga_game(game_number):
    game = Jenga()
    game.play_game()


if __name__ == "__main__":
    start_time = datetime.now()
    total_games = 10000

    threads = []
    for game_number in range(total_games):
        thread = threading.Thread(target=run_jenga_game, args=(game_number,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()

    print(f"Total time taken for {total_games} games (Multithreading): {total_time} seconds")
