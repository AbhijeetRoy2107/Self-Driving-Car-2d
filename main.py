import os
import train

WINNER_FILE = "winner_genome.pkl"

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_file.txt")
    winner_path = os.path.join(local_dir, WINNER_FILE)

    train.run_training(config_path, winner_path=winner_path, generations=10000)
    print(f"Training done. Best genome saved to: {winner_path}")
