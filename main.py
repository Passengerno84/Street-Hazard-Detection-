import argparse
from train import train
from test import test

def train_fn():
    print("Training started...")
    train()
    print("Training finished!")

def test_fn():
    print("Testing started...")
    test()
    print("Testing finished!")

def main():
    parser = argparse.ArgumentParser(description="Run training or testing pipeline.")
    parser.add_argument(
        "--train", action="store_true", help="Run training mode"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run testing mode"
    )
    args = parser.parse_args()

    if args.train:
        train_fn()
    elif args.test:
        test_fn()
    else:
        print("⚠️ Please provide --train or --test")

if __name__ == "__main__":
    main()