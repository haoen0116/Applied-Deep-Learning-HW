import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from datasets import Anime, Shuffler
import torchvision.transforms as Transform

os.makedirs("images/static/", exist_ok=True)
os.makedirs("images/varying_c1/", exist_ok=True)
os.makedirs("images/varying_c2/", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', help='Device to train the model on',
                    default='cuda', choices=['cuda', 'cpu'], type=str)
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code")
parser.add_argument("--n_classes", type=int, default=15, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser,add_argument("--root_dir", type=str, required=True)
opt = parser.parse_args()
print(opt)

random_sample_dir = 'random_generation'
fixed_attribute_dir = 'fixed_attributes'
checkpoint_dir = 'checkpoints'

if not os.path.exists(random_sample_dir):
    os.makedirs(random_sample_dir)
if not os.path.exists(fixed_attribute_dir):
    os.makedirs(fixed_attribute_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

cuda = True if torch.cuda.is_available() else False

if opt.device == 'cuda' and not torch.cuda.is_available():
    print("Your device currenly doesn't support CUDA.")
    exit()
print('Using device: {}'.format(opt.device))


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code


# Loss functions
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()  # original for label
continuous_loss = torch.nn.MSELoss()
# criterion = torch.nn.BCELoss()
criterion = torch.nn.MSELoss()
# Loss weights
lambda_cat = 1
lambda_con = 0.1

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    categorical_loss.cuda()
    continuous_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(opt.b1, opt.b2))  # opt.lr
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(opt.b1, opt.b2))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=0.0001, betas=(opt.b1, opt.b2)
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Static generator inputs for sampling
static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
static_label = to_categorical(
    np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
)
static_code = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Static sample
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    print("static_label::::::::", static_label.size())
    static_sample = generator(z, static_label, static_code)
    save_image(static_sample.data, "images/static/%d.png" % batches_done, nrow=n_row, normalize=True)

    # Get varied c1 and c2
    zeros = np.zeros((n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
    c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))
    sample1 = generator(static_z, static_label, c1)
    sample2 = generator(static_z, static_label, c2)
    save_image(sample1.data, "images/varying_c1/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(sample2.data, "images/varying_c2/%d.png" % batches_done, nrow=n_row, normalize=True)


def get_random_label(batch_size, hair_classes, eye_classes, face_classes, glasses_classes):
    """ Sample a batch of random class labels given the class priors.

    Args:
        batch_size: number of labels to sample.
        hair_classes: number of hair colors.
        hair_prior: a list of floating points values indicating the distribution
					      of the hair color in the training data.
        eye_classes: (similar as above).
        eye_prior: (similar as above).

    Returns:
        A tensor of size N * (hair_classes + eye_classes).
    """

    hair_code = torch.zeros(batch_size, hair_classes)  # One hot encoding for hair class
    # print('hair_code', hair_code.size())
    eye_code = torch.zeros(batch_size, eye_classes)  # One hot encoding for eye class
    face_code = torch.zeros(batch_size, face_classes)
    glasses_code = torch.zeros(batch_size, glasses_classes)

    hair_type = np.random.choice(hair_classes, batch_size)  # Sample hair class from hair class prior
    eye_type = np.random.choice(eye_classes, batch_size)  # Sample eye class from eye class prior
    face_type = np.random.choice(face_classes, batch_size)  # Sample eye class from eye class prior
    glasses_type = np.random.choice(glasses_classes, batch_size)  # Sample eye class from eye class prior

    for i in range(batch_size):
        hair_code[i][hair_type[i]] = 1
        eye_code[i][eye_type[i]] = 1
        face_code[i][face_type[i]] = 1
        glasses_code[i][glasses_type[i]] = 1
    # print('666hair_code', hair_code.size())

    return torch.cat((hair_code, eye_code, face_code, glasses_code), dim=1)


def denorm(img):
    """ Denormalize input image tensor. (From [0,1] -> [-1,1])

    Args:
        img: input image tensor.
    """

    output = img / 2 + 0.5
    return output.clamp(0, 1)


def save_model(model, optimizer, step, log, file_path):
    """ Save model checkpoints. """

    state = {'model': model.state_dict(),
             'optim': optimizer.state_dict(),
             'step': step,
             'log': log}
    torch.save(state, file_path)
    return


# ----------
#  Dataloader
# ----------
# root_dir = '../selected_cartoonset100k/'
root_dir = args.root_dir

transform = Transform.Compose([Transform.ToTensor(),
                               Transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = Anime(root_dir=root_dir, transform=transform)
shuffler = Shuffler(dataset=dataset, batch_size=opt.batch_size)
device = opt.device

# ----------
#  Training
# ----------

# for epoch in range(opt.n_epochs):
#     for i, (imgs, labels) in enumerate(dataloader):
d_log, g_log, info_log = [], [], []

for step_i in range(1, 300000 + 1):
    real_img, hair_tags, eye_tags, face_tags, glasses_tags = shuffler.get_batch()
    # real_img, hair_tags, eye_tags, face_tags, glasses_tags = real_img.to(device), \
    #                                                          hair_tags.to(device), \
    #                                                          eye_tags.to(device), \
    #                                                          face_tags.to(device), \
    #                                                          glasses_tags.to(device)
    batch_size = real_img.shape[0]  # torch.Size([64, 3, 128, 128])

    # Adversarial ground truths
    valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
    # print(labels.size(),labels)
    # Configure input
    real_imgs = Variable(real_img.type(FloatTensor))
    # labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

    # -----------------
    #  Train Generator
    # -----------------

    optimizer_G.zero_grad()

    # Sample noise and labels as generator input
    z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
    # label_input = get_random_label(batch_size=batch_size,
    #                             hair_classes=6,
    #                             eye_classes=4,
    #                             face_classes=3,
    #                             glasses_classes=2).to(device)
    label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
    code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))
    # Generate a batch of images
    gen_imgs = generator(z, label_input, code_input)
    # Loss measures generator's ability to fool the discriminator
    validity, _, _ = discriminator(gen_imgs)  # validity [64, 1]
    g_loss = adversarial_loss(validity, valid)  # valid is all 1
    g_log.append(g_loss.item())
    g_loss.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Loss for real images
    real_pred, _, _ = discriminator(real_imgs)
    d_real_loss = adversarial_loss(real_pred, valid)

    # Loss for fake images
    fake_pred, _, _ = discriminator(gen_imgs.detach())
    d_fake_loss = adversarial_loss(fake_pred, fake)

    # Total discriminator loss
    d_loss = (d_real_loss + d_fake_loss) / 2
    d_log.append(d_loss.item())
    d_loss.backward()
    optimizer_D.step()

    # ------------------
    # Information Loss
    # ------------------

    optimizer_info.zero_grad()

    # Sample labels
    sampled_labels = np.random.randint(0, opt.n_classes, batch_size)  # [64,_]

    # Ground truth labels
    gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

    # Sample noise, labels and code as generator input
    z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
    label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
    code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

    gen_imgs = generator(z, label_input, code_input)
    _, pred_label, pred_code = discriminator(gen_imgs)
    # print("pred_label",pred_label.size(), "gt_labels", gt_labels.size())
    info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
        pred_code, code_input
    )

    info_loss.backward()
    optimizer_info.step()

    # --------------
    # Log Progress
    # --------------
    print("[step_i %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
          % (step_i, 50000, d_loss.item(), g_loss.item(), info_loss.item())
          )
    # print(
    #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
    #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
    # )
    # batches_done = epoch * len(dataloader) + i
    # if step_i % 20 == 0:
    #     sample_image(n_row=10, batches_done=step_i)
    ########## Checkpointing ##########
    if step_i == 1:
        save_image(denorm(real_img[:32, :, :, :]), os.path.join(random_sample_dir, 'real.png'))
    if step_i % 50 == 0:
        save_image(denorm(gen_imgs[:32, :, :, :]),
                   os.path.join(random_sample_dir, 'fake_step_{}.png'.format(step_i)))
    if step_i % 200 == 0:
        save_model(model=generator, optimizer=optimizer_G, step=step_i, log=tuple(g_log),
                   file_path=os.path.join(checkpoint_dir, 'G_{}.ckpt'.format(step_i)))
        save_model(model=discriminator, optimizer=optimizer_D, step=step_i, log=tuple(d_log),
                   file_path=os.path.join(checkpoint_dir, 'D_{}.ckpt'.format(step_i)))
