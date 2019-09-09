import torch
import torch.nn
import torchvision.transforms as Transform
from torchvision.utils import save_image
from argparse import ArgumentParser
import utils
from datasets_test import Anime
from ACGAN_split_128size import Generator
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--device', help='Device to train the model on',
                    default='cuda', choices=['cuda', 'cpu'], type=str)
parser.add_argument('--label_dir', help='Label direction', type=str, required=True)
parser.add_argument('--output_dir', help='Output direction', type=str, required=True)
args = parser.parse_args()

if args.device == 'cuda' and not torch.cuda.is_available():
    device = 'cpu'
else:
    device = 'cuda'

def main():
    latent_dim = 200
    hair_classes, eye_classes, face_classes, glasses_classes = 6, 4, 3, 2
    num_classes = hair_classes + eye_classes + face_classes + glasses_classes

    G_path = './checkpoint/ACGAN_generator_FID166.ckpt'
    prev_state = torch.load(G_path, map_location=device)

    G = Generator(latent_dim=latent_dim, class_dim=num_classes).to(device)
    G.load_state_dict(prev_state['model'])
    G = G.eval()

    # root_dir = '../sample_test/sample_fid_testing_labels.txt'
    root_dir = args.label_dir
    transform = Transform.Compose([Transform.ToTensor(),
                                   Transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = Anime(root_dir=root_dir, transform=transform, batch_size=64)

    for i in tqdm(range(dataset.length())):
        # print("Step: ", i)
        hair_tags, eye_tags, face_tags, glasses_tags = dataset.get_item(i)
        hair_tags, eye_tags, face_tags, glasses_tags = hair_tags.to(device), \
                                                       eye_tags.to(device), \
                                                       face_tags.to(device), \
                                                       glasses_tags.to(device)

        z = torch.randn(1, latent_dim).to(device)
        fake_tag = torch.cat((hair_tags, eye_tags, face_tags, glasses_tags)).unsqueeze(0).float().to(device)
        fake_img = G(z, fake_tag).to(device)
        save_image(utils.denorm(fake_img), '{}/{}.png'.format(args.output_dir, i))


if __name__ == '__main__':
    main()
