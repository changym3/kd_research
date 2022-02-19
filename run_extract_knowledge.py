import argparse

import kd.knowledge as K


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='./examples/ckpt/test_GAT/')
    args = parser.parse_args()

    K.extract_and_save_knowledge(args.ckpt_dir)
    