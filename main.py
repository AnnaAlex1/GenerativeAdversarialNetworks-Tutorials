import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


# Things to try:
# 1. What happens if you use larger network?
# 2. Better normalization with BatchNorm
# 3. Different learning rate (is there a beeter one?)
# 4. Change the architecture to a CNN


# Discriminator

class Discriminator(nn.Module):                 # inherits from nn.Module

    # constructor
    def __init__(self, img_dim):
        super().__init__()

        #build a simple model
        self.disc == nn.Sequential(
            nn.Linear(img_dim, 128),         # input size = in_features, output size = 128
            nn.LeakyReLU(0.1),                    # (negative_slope=0.1)
            nn.Linear(128, 1),                    # we only want one value for discriminator (real or not)
            nn.Sigmoid(),                         # we call Sigmoid on the last layer
        )

        def forward(self, x):
            return self.disc(x)



# Generator

class Generator(nn.Module):                 # inherits from nn.Module

    # constructor
    def __init__(self, z_dim, img_dim):                    # z_dim -> the dimension of the noise/latent noise that the generator will take as input
        super().__init__()

        #build a simple model
        self.gen == nn.Sequential(
            nn.Linear(z_dim, 256), 
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),            # img_dim = 784 because the MNIST dataset is 28*28*1
            nn.Sigmoid(),                         # we call Sigmoid on the last layer
            nn.Tanh(),                              # we normalize output between (-1,1) because input is also normalized between (-1,1)
        )

        def forward(self, x):
            return self.gen(x)




# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4   # learning rate - important! - play around
z_dim = 64      # 32, 128, 256
image_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size,z_dim)).to(device)    # to see how it changed with epochs             
                                                            # torch.randn() Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.normalize((0.5,), (0,5,))]             # transforms image to tensor and normalizes values
)
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")                   #######
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")     
step = 0 # for tensorboard

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):          # _ in the place of label to do unsupervised learning
        real = real.view(-1, 784).to(device)                  # for reshaping: returns a tensor with the new shape. The returned tensor will share the underling data with the original tensor
        batch_size = real.shape[0]      # first dimension is our batch_size

        ### TRAIN DISCRIMINATOR: max log(D(real)) + log(1 - D(G(z)))   # z -> noise
        noise = torch.randn((batch_size, z_dim)).to(device)
        fake = gen(noise)           # generate fake images
        
        # for the first part of the expression
        disc_real =   disc(real).view(-1)          # log(D(real)) -> what the discriminator outputs on real ones, .view(-1) -> flattens everything
        lossD_real = criterion(disc_real, torch.ones_like())  # ones so the second part of the equation = 0

        # for the second part of the expression
        disc_fake = disc(fake.detatch()).view(-1) 
        # disc_fake = disc(fake.detatch()).view(-1)           # use detatch so lossD.backward() does not clear the intermediate computations from cache (one of the two options)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))  # zeros, so the first part disappears

        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)  # same purpose as detatch() above (second of two options)
        opt_disc.step()


        ### TRAIN GENERATOR: min log(1 - D(G(z)))   <-> max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()            # The call to loss.backward() computes the partial derivative of the output f with respect to each of the input variables.
        opt_gen.step()

     
        # for the second part of the expression
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))  # zeros, so the first part disappears

        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()


        # Additional code for tensorboard
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1