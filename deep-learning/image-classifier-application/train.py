import argparse
import sys
import torch
from network import *

def get_args(argv=None):
    parser = argparse.ArgumentParser(
            description='training a network'
            )

    parser.add_argument('data_dir', action='store', help='Data directory')
    parser.add_argument('--save_dir', action='store', help='Set directory to save checkpoints')
    parser.add_argument('--arch', action='store', default='densenet121', help='Choose Pre-trained network')
    parser.add_argument('--learning_rate', action='store', default=0.01, type=float, help='Hyperparameters: learning_rate')
    parser.add_argument('--output_size', action='store', default=102, type=int, help='Output size')
    parser.add_argument('--epochs', action='store', default=1, type=int, help='Hyperparameters: epochs')
    parser.add_argument('--hidden_units', action='store', nargs='+', type=int, help='Hyperparameters: hidden units')
    parser.add_argument('--drop_p', action='store', default=0.5, type=float, help='Hyperparameters: dropout probility')
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
        
    # load data
    print('--> Loading the data .. ')
    image_datasets, dataloaders = load_data(args.data_dir)
        
    # build model
    print('--> Building the model .. ')
    model = Network(args.drop_p, args.hidden_units, args.output_size, image_datasets['train'].class_to_idx, args.arch)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)    
    
    # training the model
    print('--> Training the model .. ')
    with active_session():
        train(model, dataloaders['train'], dataloaders['valid'], criterion=criterion, 
            optimizer=optimizer, epochs=args.epochs, print_every=30, device=device)
        
    # do validation on the test set
    print('--> Testing the model .. ')
    with active_session():
        validation(model, dataloaders['test'], criterion, device=device)
    
    # save the checkpoint
    print('--> Saving the checkpoint .. ')
    checkpoint = {
    'drop_p': args.drop_p,
    'hidden_layer_size': [each.out_features for each in model.classifier.hidden_layers],
    'output_size': args.output_size,
    'learning_rate': args.learning_rate,
    'arch': args.arch,
    'epoch': args.epoch,
    'class_to_idx': model.class_to_idx,    
    'state_dict': model.state_dict()
    }
    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')    
    
