import argparse

import kd.knowledge as K


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='./examples/ckpt/test_GAT/')
    parser.add_argument('--model_name', type=str, default='GNN')
    args = parser.parse_args()

    K.extract_and_save_knowledge(args.ckpt_dir, model_name=args.model_name)
    