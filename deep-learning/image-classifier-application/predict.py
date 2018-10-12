import argparse
import sys
import torch
from os.path import isfile
from network import *

def get_args(argv=None):
    parser = argparse.ArgumentParser(
            description='training a network'
            )

    parser.add_argument('image_path', action='store', help='A single image')
    parser.add_argument('checkpoint', action='store', help='Checkpoint for pre-trained model')
    parser.add_argument('--top_k', action='store', default=5, type=int, help='Top K most likely classes')
    parser.add_argument('--category_names', action='store', default='cat_to_name.json', help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU for training')

    return parser.parse_args(argv)

if __name__ == '__main__':
    # get arguments
    args = get_args(sys.argv[1:])
    print('--> arguments: {}'.format(args))

    # config cpu or gpu mode
    if args.gpu:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            print("----> Note: gpu mode is not available, change to cpu mode")
            device = 'cpu'
    else:
        device = 'cpu'
    print('--> Config mode: {}'.format(device))
    
    # Label mapping 
    if not isfile(args.category_names):
        print('{} is not a valid file'.format(args.category_names))
        sys.exit()
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # predict a image   
    print('--> Predicting a image .. ')
    if not isfile(args.checkpoint):
        print('{} is not a valid file'.format(args.checkpoint))
        sys.exit()
    probs, classes = predict(args.image_path, args.checkpoint, device, args.top_k)
    
    # check sanity
    print('--> Checking sanity .. ')    
    # Label mapping 
    if not isfile(args.category_names):
        print('{} is not a valid file'.format(args.category_names))
        sys.exit()
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    check_sanity(args.image_path, probs, classes, cat_to_name)
