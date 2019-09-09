import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms as Transform
from torchvision.utils import save_image

import numpy as np
import os
from argparse import ArgumentParser
import utils
from datasets_test import Anime
from ACGAN_split_128size import Generator, Discriminator
from utils import save_model, denorm, plot_loss, plot_classifier_loss, show_process
from utils import generation_by_attributes, get_random_label

parser = ArgumentParser()
parser.add_argument('-d', '--device', help='Device to train the model on',
                    default='cuda', choices=['cuda', 'cpu'], type=str)
parser.add_argument('-i', '--iterations', help='Number of iterations to train ACGAN',
                    default=50000, type=int)
parser.add_argument('-b', '--batch_size', help='Training batch size',
                    default=64, type=int)
parser.add_argument('-t', '--train_dir', help='Training data directory',
                    default='../data', type=str)
parser.add_argument('-s', '--sample_dir', help='Directory to store generated images',
                    default='./samples', type=str)
parser.add_argument('-c', '--checkpoint_dir', help='Directory to save model checkpoints',
                    default='./checkpoints', type=str)
parser.add_argument('--sample', help='Sample every _ steps',
                    default=20, type=int)
parser.add_argument('--check', help='Save model every _ steps',
                    default=200, type=int)
parser.add_argument('--lr', help='Learning rate of ACGAN. Default: 0.0002',
                    default=0.0001, type=float)
parser.add_argument('--beta', help='Momentum term in Adam optimizer. Default: 0.5',
                    default=0.5, type=float)
parser.add_argument('--classification_weight', help='Classification loss weight. Default: 1',
                    default=1, type=float)
parser.add_argument('--model_dir', help='Folder where the trained model is saved',
                    default='../models', type=str)
args = parser.parse_args()

if args.device == 'cuda' and not torch.cuda.is_available():
    print("Your device currenly doesn't support CUDA.")
    exit()
print('Using device: {}'.format(args.device))


def main():
    if not os.path.exists('Human_result'):
        os.makedirs('Human_result')

    latent_dim = 200
    device = args.device
    hair_classes, eye_classes, face_classes, glasses_classes = 6, 4, 3, 2
    num_classes = hair_classes + eye_classes + face_classes + glasses_classes

    G_path = '{}/ACGAN_generator.ckpt'.format(args.model_dir)
    prev_state = torch.load(G_path)

    G = Generator(latent_dim=latent_dim, class_dim=num_classes).to(device)
    G.load_state_dict(prev_state['model'])
    G = G.eval()

    root_dir = '../sample_test/sample_human_testing_labels.txt'
    transform = Transform.Compose([Transform.ToTensor(),
                                   Transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = Anime(root_dir=root_dir, transform=transform, batch_size=1)

    for i in range(dataset.length()):
        print("Step: ", i)
        hair_tags, eye_tags, face_tags, glasses_tags = dataset.get_item(i)
        hair_tags, eye_tags, face_tags, glasses_tags = hair_tags.to(device), \
                                                       eye_tags.to(device), \
                                                       face_tags.to(device), \
                                                       glasses_tags.to(device)


        z = torch.randn(1, latent_dim).to(device)
        fake_tag = torch.cat((hair_tags, eye_tags, face_tags, glasses_tags)).unsqueeze(0).float().to(device)
        # print('z', z.size())    # [128, 100]
        # print('fake_tag', fake_tag.size())  # [128, 15]
        fake_img = G(z, fake_tag).to(device)
        save_image(utils.denorm(fake_img), 'Human_result/{}.png'.format(i))
        # print(fake_img.size())  # [128, 3, 64, 64]


if __name__ == '__main__':
    main()
