import argparse

from kd.knowledge import extract_and_save_knowledge


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='./examples/ckpt/test_GAT/')
    args = parser.parse_args()

    extract_and_save_knowledge(args.ckpt_dir)
    