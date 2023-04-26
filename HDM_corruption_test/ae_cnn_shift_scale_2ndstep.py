import  torch
from    torch import nn
from    torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

device = torch.device('cuda')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=2048): #1024
        return input.view(input.size(0), size, 1) #input.view(input.size(0), size, 1, 1)

class Encoder(nn.Module):

    def __init__(self, imgsz, n_hidden1, n_output, keep_prob):
        super(Encoder, self).__init__()

        self.imgsz = imgsz
        self.n_hidden = 2048 #1024
        self.n_output = n_output
        self.keep_prob = keep_prob

        self.conv1 = nn.Conv1d(93, 24, kernel_size=4, stride=2, padding=0,
                               bias=False)
        self.conv2 = nn.Conv1d(24, 48, kernel_size=4, stride=2, padding=0,
                               bias=False)
        self.conv3 = nn.Conv1d(48, 96, kernel_size=4, stride=2, padding=0,
                               bias=False)
        self.conv4 = nn.Conv1d(96, 192, kernel_size=4, stride=2, padding=0,
                               bias=False)
        #self.flatten = Flatten()


        self.in_dim = 64 #32

        #100, 49, 23, 10, 4 - f4
        self.net1 = nn.Sequential(

            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=0), #16
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=0), #8
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2), #4
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2),  #256 n_output #h_dim=1024 #2
            nn.ReLU(),
            Flatten(),

        )

        self.net2 = nn.Sequential(

            nn.Conv1d(93, self.in_dim, kernel_size=4, stride=2), #Conv2d #ecg 140x1
            nn.BatchNorm1d(self.in_dim),
            #nn.ReLU(),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.in_dim, self.in_dim*2, kernel_size=4, stride=2),
            #nn.ReLU(),
            nn.BatchNorm1d(self.in_dim*2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.in_dim*2, self.in_dim*4, kernel_size=4, stride=2),
            #nn.ReLU(),
            nn.BatchNorm1d(self.in_dim*4),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.in_dim*4, self.in_dim*8, kernel_size=4, stride=2),  #256 n_output #h_dim=1024
            #nn.ReLU(),
            nn.BatchNorm1d(self.in_dim*8),
            nn.LeakyReLU(0.1),
            Flatten(),

        )

        self.net = nn.Sequential(

            nn.Conv1d(32, 32, kernel_size=4, stride=2), #Conv2d #ecg 140x1
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2),  #256 n_output #h_dim=1024
            nn.ReLU(),
            Flatten(),

        )

        self.fc1 = nn.Linear(2048, n_output) # h_dim=1024 = 4x256, z_dim=32
        self.fc2 = nn.Linear(2048, n_output) #1792 = 7x256 #1536

        self.fc3 = nn.Sequential(nn.Linear(2048, n_output),
            nn.BatchNorm1d(n_output),
            nn.LeakyReLU(0.1),
        )


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() #sigma
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]


    def forward(self, x):
        """

        :param x:
        :return:
        """
        h = self.net2(x) #mu_sigma

        h = self.fc1(h)

        return h #mu, std


class Decoder(nn.Module):


    def __init__(self, dim_z, n_hidden1, n_output, keep_prob):
        super(Decoder, self).__init__()

        self.n_hidden = 2048 #1024
        self.dim_z = dim_z
        self.n_output = n_output
        self.keep_prob = keep_prob

        #self.fc3 = nn.Linear(dim_z, self.n_hidden) #20
        self.fc3 = nn.Sequential(nn.Linear(dim_z, self.n_hidden),
            nn.BatchNorm1d(self.n_hidden),
            nn.LeakyReLU(0.1),
        )



        self.out_dim = 64

        #140 69 34 16 7 - 3,2
        #140 69 33 15 6 - 4,2

        #1 5 14 32 68 140
        self.dec1 = nn.ConvTranspose1d(self.n_hidden, 256, kernel_size=4, stride=2)
        self.dec2 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2)
        self.dec3 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2) #,padding=1
        self.dec4 = nn.ConvTranspose1d(64, 32, kernel_size=6, stride=2)
        self.dec5 = nn.ConvTranspose1d(32, 93, kernel_size=6, stride=2)


        #1st
        self.net1 = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose1d(self.n_hidden, self.out_dim*8, kernel_size=4, stride=2), #ConvTranspose2d
            #nn.ReLU(),
            nn.BatchNorm1d(self.out_dim*8),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(self.out_dim*8, self.out_dim*4, kernel_size=4, stride=2),
            #nn.ReLU(),
            nn.BatchNorm1d(self.out_dim*4),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(self.out_dim*4, self.out_dim*2, kernel_size=4, stride=2),
            #nn.ReLU(),
            nn.BatchNorm1d(self.out_dim*2),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(self.out_dim*2, self.out_dim, kernel_size=6, stride=2),
            #nn.ReLU(),
            nn.BatchNorm1d(self.out_dim),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(self.out_dim, 93, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

        self.net2 = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose1d(self.n_hidden, 256, kernel_size=5, stride=2), #ConvTranspose2d
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )



        self.net = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose1d(self.n_hidden, 128, kernel_size=5, stride=2), #ConvTranspose2d
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )


    def forward(self, h):
        """

        :param h:
        :return:
        """
        h = self.fc3(h)


        result = self.net1(h)

        return result





def init_weights(encoder, decoder):

    def init_(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    for m in encoder.modules():
        m.apply(init_)
    for m in decoder.modules():
        m.apply(init_)

    print('weights inited!')



def get_ae(encoder, decoder, x):
    # encoding
    a = encoder(x)

    # decoding
    y = decoder(a)
    y = torch.clamp(y, 1e-8, 1 - 1e-8) 

    return y



def get_z(encoder, x):

    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    return z


def get_loss_ae(encoder, decoder, x, x_target):
    """

    :param encoder:
    :param decoder:
    :param x: input
    :param x_hat: target
    :param dim_img:
    :param dim_z:
    :param n_hidden:
    :param keep_prob:
    :return:
    """
    batchsz = x.size(0)

    # encoding
    h = encoder(x)

    # decoding
    y = decoder(h)
    y = torch.clamp(y, 1e-8, 1 - 1e-8)

    marginal_likelihood = -nn.MSELoss(reduction='sum')(y, x_target) / batchsz


    KL_divergence = 0.0 * torch.sum(torch.pow(y, 2))

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return y, ELBO, loss, marginal_likelihood, KL_divergence


def get_loss(encoder, decoder, x, x_target):
    """

    :param encoder:
    :param decoder:
    :param x: input
    :param x_hat: target
    :param dim_img:
    :param dim_z:
    :param n_hidden:
    :param keep_prob:
    :return:
    """
    batchsz = x.size(0)

    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    y = torch.clamp(y, 1e-8, 1 - 1e-8) #1e-8, 1 - 1e-8

    marginal_likelihood = -nn.MSELoss(reduction='sum')(y, x_target) / batchsz

    KL_divergence = 0.5 * torch.sum(
                                torch.pow(mu, 2) +
                                torch.pow(sigma, 2) -
                                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                               ).sum() / batchsz

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return y, z, loss, marginal_likelihood, KL_divergence

def get_loss2(encoder, decoder, x, x_target):
    """

    :param encoder:
    :param decoder:
    :param x: input
    :param x_hat: target
    :param dim_img:
    :param dim_z:
    :param n_hidden:
    :param keep_prob:
    :return:
    """
    batchsz = x.size(0)

    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    y = torch.clamp(y, 1e-8, 1 - 1e-8) #1e-8, 1 - 1e-8

    # encoding
    mu2, sigma2 = encoder(y)
    # sampling by re-parameterization technique
    z2 = mu2 + sigma2 * torch.randn_like(mu2)


    marginal_likelihood = -nn.MSELoss(reduction='sum')(y, x_target) / batchsz

    KL_divergence = 0.25 * torch.sum(
                                torch.pow(mu, 2) +
                                torch.pow(sigma, 2) -
                                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                               ).sum() / batchsz

    KL_divergence2 = 0.25 * torch.sum(
                                torch.pow(mu2, 2) +
                                torch.pow(sigma2, 2) -
                                torch.log(1e-8 + torch.pow(sigma2, 2)) - 1
                               ).sum() / batchsz

    ELBO = marginal_likelihood - KL_divergence - KL_divergence2

    loss = -ELBO

    return y, z, loss, marginal_likelihood, KL_divergence



def imq_kernel(X,Y,h_dim, b_size):

    batch_size = b_size

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats




def get_loss_wae(encoder, decoder, x, x_target):
    """

    :param encoder:
    :param decoder:
    :param x: input
    :param x_hat: target
    :param dim_img:
    :param dim_z:
    :param n_hidden:
    :param keep_prob:
    :return:
    """
    batchsz = x.size(0)

    # encoding
    mu, sigma = encoder(x)
    # decoding
    y = decoder(mu)
    y = torch.clamp(y, 1e-8, 1 - 1e-8)

    recon_loss = nn.MSELoss(reduction='sum')(y, x_target) / batchsz


    # ======== MMD Kernel Loss ======== #

    batch_size = batchsz

    z_fake = Variable(torch.randn(x.size()[0], encoder.n_output) * 1)

    if torch.cuda.is_available():
        z_fake = z_fake.cuda()

        z_real,std = encoder(x)

        mmd_loss = imq_kernel(z_real, z_fake, h_dim=encoder.n_output, b_size=batchsz)
        mmd_loss = mmd_loss / batch_size

        total_loss = recon_loss + mmd_loss



    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)


    return y, z, total_loss, recon_loss, mmd_loss

def get_loss_wae2(encoder, decoder, x, x_target):
    """

    :param encoder:
    :param decoder:
    :param x: input
    :param x_hat: target
    :param dim_img:
    :param dim_z:
    :param n_hidden:
    :param keep_prob:
    :return:
    """
    batchsz = x.size(0)

    # encoding
    mu, sigma = encoder(x)
    # decoding
    y = decoder(mu)
    y = torch.clamp(y, 1e-8, 1 - 1e-8)

    recon_loss = nn.MSELoss(reduction='sum')(y, x_target) / batchsz

    # ======== MMD Kernel Loss ======== #

    batch_size = batchsz

    z_fake = Variable(torch.randn(x.size()[0], encoder.n_output) * 1) #n_z = 8, sigma = 1

    if torch.cuda.is_available():
        z_fake = z_fake.cuda()

        z_real,std = encoder(x)
        yt = decoder(z_real)
        z_realt,stdt = encoder(yt)

        mmd_loss = imq_kernel(z_real, z_fake, h_dim=encoder.n_output, b_size=batchsz)
        mmd_loss = mmd_loss / batch_size

        mmd_loss2 = imq_kernel(z_realt, z_fake, h_dim=encoder.n_output, b_size=batchsz)
        mmd_loss2 = mmd_loss2 / batch_size

        mmd_losst = 0.5*mmd_loss + 0.5*mmd_loss2

        total_loss = recon_loss + mmd_losst


    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)


    return y, z, total_loss, recon_loss, mmd_losst




def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


def warp(poses, gamma):
	'''
	Input poses is of shape batch_size x channels (3x25x2) x seq_len 
	'''

	batch_size, num_channels, seq_len = poses.size()

	#poses = poses.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_joints*num_channels)

	poses = poses.permute(0,2,1)
	pose_vec_len = num_channels#*num_joints
	max_gamma = seq_len - 1

	gamma_0 = torch.floor(gamma)
	gamma_0 = gamma_0.int()
	gamma_1 = gamma_0 + 1

	gamma_0 = torch.clamp(gamma_0, 0, max_gamma)
	gamma_1 = torch.clamp(gamma_1, 0, max_gamma)

	gamma_expand = gamma.unsqueeze(2)
	gamma_0_expand = gamma_0.unsqueeze(2)
	gamma_1_expand = gamma_1.unsqueeze(2)

	gamma_tile = gamma_expand.repeat(1,1,pose_vec_len)
	gamma_0_tile = gamma_0_expand.repeat(1,1,pose_vec_len)
	gamma_1_tile = gamma_1_expand.repeat(1,1,pose_vec_len)

	poses_flat = poses.contiguous().view(batch_size*seq_len, pose_vec_len)
	gamma_0_flat = gamma_0.view(batch_size*seq_len)
	gamma_0_flat = gamma_0_flat.long()
	gamma_1_flat = gamma_1.view(batch_size*seq_len)
	gamma_1_flat = gamma_1_flat.long()

	range_vec = Variable(torch.arange(0,batch_size), requires_grad=False)
	range_vec = range_vec.cuda()  #.cuda()
	range_vec = range_vec.unsqueeze(1)
	range_vec_tile =  range_vec.repeat(1, seq_len)
	range_vec_tile_vec = range_vec_tile.view(batch_size*seq_len)
	offset = range_vec_tile_vec*seq_len
	offset = offset.long()

	add0 = gamma_0_flat + offset
	add1 = gamma_1_flat + offset

	add0 = add0.repeat(pose_vec_len, 1)
	add1 = add1.repeat(pose_vec_len, 1)

	add0 = torch.t(add0)
	add1 = torch.t(add1)

	add0 = add0.detach()
	add1 = add1.detach()

	Ia_flat = torch.gather(poses_flat, 0 , add0)
	Ib_flat = torch.gather(poses_flat, 0 , add1)

	Ia = Ia_flat.view(batch_size, seq_len, pose_vec_len)
	Ib = Ib_flat.view(batch_size, seq_len, pose_vec_len)

	gamma_0_tile = gamma_0_tile.float()
	gamma_1_tile = gamma_1_tile.float()

	wa = 1 - (gamma_tile - gamma_0_tile)
	wb = 1 - wa

	output = wa*Ia + wb*Ib

	output = output.permute(0,2,1)

	return output


class SurrogateNet2(nn.Module):
    def __init__(self):
        super(SurrogateNet2, self).__init__()

        self.first_layer_init = None
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init)}
        nl, nl_weight_init, first_layer_init = nls_and_inits['sine']

        #if weight_init is not None:  # Overwrite weight init if passed
        #    self.weight_init = weight_init
        #else:
        self.weight_init = nl_weight_init


        self.conv1 = nn.Conv1d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7),
            nn.MaxPool1d(2, stride=2),
            nn.ReLU(True),
            #nl,
            nn.Conv1d(8, 10, kernel_size=5),
            nn.MaxPool1d(2, stride=2),
            #nl
            nn.ReLU(True)
        )

        self.localization2 = nn.Sequential(
            #nn.Conv1d(93, 1, kernel_size=1),
            #nn.ReLU(True),
            nn.Conv1d(93, 32, kernel_size=7),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
            nn.ReLU(True),
            #nl,
            nn.Conv1d(32, 64, kernel_size=5),
            nn.MaxPool1d(2, stride=2),
            #nl
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            #nn.Linear(10 * 3 * 3, 32),
            nn.Linear(10 * 31, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )


        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))




        self.net1 = nn.Sequential(

            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=0), #16 #5, #2
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, stride=1, padding=0), #8 #5
            nn.ReLU(),
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=1), #6, #2
            nn.Tanh(),
            nn.ConvTranspose1d(8, 1, kernel_size=5, stride=1), #6
            nn.Tanh(), 
            nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0),
        )



        self.net2 = nn.Sequential(
            nn.Sigmoid(),
        )

        self.fc_net1 = nn.Sequential(
            nn.Linear(10 * 996, 512),
            nl, 
            nn.Linear(512, 64),
            nl, 
            nn.Linear(64, 512),
            nl,
            nn.Linear(512, 100 * 40)
        )

        self.fc_net1a = nn.Sequential(
            nn.Linear(10 * 996, 1024),
            nl, 
            nn.Linear(1024, 512),
            nl, 
            nn.Linear(512, 1024),
            nl,
            nn.Linear(1024, 10 * 400)
        )

        self.fc_net_a = nn.Sequential( 
            nn.Linear(10 * 2321, 10*728),
            nl, 
            nn.Linear(10*728, 10*512),
            nl, 
            nn.Linear(10*512, 100 * 93)
        )

        self.fc_net_b = nn.Sequential( 
            nn.Linear(10 * 2321, 10*512),
            nl, 
            nn.Linear(10*512, 10*128),
            nl, 
            nn.Linear(10*128, 100 * 93)
        )

        self.fc_net = nn.Sequential( 
            nn.Linear(10 * 3251, 10*512),
            nl, 
            nn.Linear(10*512, 10*96), 
            nl, 
            nn.Linear(10*96, 140 * 93)
        )

        self.fc_netaa = nn.Sequential( 
            nn.Linear(10 * 2321, 10*1024),
            nl, 
            nn.Linear(10*1024, 10*1024),
            nl, 
            nn.Linear(10*1024, 100 * 93)
        )


        self.fc_net.apply(self.weight_init)
        self.fc_net[0].apply(first_layer_init)

        self.fc_net2 = nn.Sequential(
            nn.Linear(140, 31 * 10),
            nl,
            nn.Linear(10 * 31, 64),
            nl, 
            nn.Linear(64, 31 * 10),
            nl,
            nn.Linear(31*10, 140)
        )

        self.fc_net2.apply(self.weight_init)
        self.fc_net2[0].apply(first_layer_init)


        self.fc_loc_shift = nn.Sequential(
            nn.Linear(64 * 31, 1),
            nn.Tanh()
        )

        self.fc_loc_scale = nn.Sequential(
            nn.Linear(64 * 31, 1), 
            nn.Tanh()
        )


        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.fc_loc_scale.apply(self.weight_init)



    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x) #batch, 10, 31

        xs = xs.view(-1, 10 * 31) #10x3x3(90) -> 10x31
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        x = x.reshape(x.shape[0], 1, x.shape[2],1)

        grid = F.affine_grid(theta, x.size()) #4d
        x = F.grid_sample(x, grid)

        return x


    def shiftnet2(self, x):
        #128, 1, 140
        xs = self.localization2(x) #batch, 10, 31

        xs = xs.view(xs.shape[0], -1) #10x3x3(90) -> 10x31
        xs2 = xs
        param = self.fc_loc_shift(xs)

        param = torch.clamp(param, -1+1e-8, 1 - 1e-8)
        param = param.view(-1, 1)*15.0

        scale_param = self.fc_loc_scale(xs2)
        scale_param = 1 + scale_param.view(-1, 1) * 0.25 



        x_test = []
        for iterb in range(x.shape[0]):
            xid = np.arange(0, x.shape[2], 1)
            x_test.append(xid)
        x_test = np.array(x_test)

        x_test = torch.from_numpy(x_test).float().cuda()
        gamma = scale_param*x_test + param


        window = x.shape[2]
        gamma = gamma.clamp(0, window)
        for i in range(x.shape[0]):
            gamma[i,0] = 0
            gamma[i,window-1] = window

        x = warp(x, gamma) #batch, ch, window

        x = x.reshape(x.shape[0], 1, -1)

        return x, param, scale_param

    def linearnet(self, x):
        xs = self.localization(x) #batch, 10, 996

        xs = xs.view(-1, 10 * 3251) #10x3x3(90) -> 10x31
        xs = self.fc_net(xs)

        xs = xs.view(-1, 1, 140*93)

        x = x*F.sigmoid(xs)

        return x


    def forward(self, x):
        # transform the input

        #####
        x1 = x.reshape(x.shape[0], 140, 93) #batch, window, ch
        x = torch.transpose(x1, 2, 1)
        x, param2, scale_param = self.shiftnet2(x)

        ##### return to origin
        x2 = x.reshape(x.shape[0], 93, 140) #batch, ch, window
        x = torch.transpose(x2, 2, 1)
        x = x.reshape(x.shape[0], 1, -1)
        #####

        x = x.reshape(x.shape[0], 1, -1)

        y = self.net1(x)
        x = self.net2(y+x)
        
        x = torch.clamp(x, 1e-8, 1 - 1e-8)

        return x, param2, scale_param



class SURROGATE(object):
    def __init__(self):
        super(SURROGATE, self).__init__()


    def train_iter(self, optimizer, train_data, obj_data, encoder, decoder, surrogate_model, batch_size, steps=25, learn_rate=0.01, clamp=(0, 1)):
        """
        :param x: Inputs to perturb
        :param y: Corresponding ground-truth labels
        :param net: Network to attack
        :param attack_steps: Number of attack iterations
        :param attack_lr: Learning rate of attacker
        :param random_init: If true, uses random initialization
        :param target: If not None, attacks to the chosen class. Dimension of target should be same as labels
        :return:
        """
        n_samples = train_data.shape[0]
        decoder.zero_grad()

        total_batch = int(n_samples / batch_size)
        train_data_ = train_data.copy()
        obj_data_ = obj_data.copy()

        for epoch in range(steps): 

            # Loop over all batches
            surrogate_model.train()

            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_data_[offset:(offset + batch_size), :]
                batch_xs_target = obj_data_[offset:(offset + batch_size), :]

                #print("batch_xs_input",batch_xs_input.shape,offset + batch_size)

                batch_xs_input = batch_xs_input.reshape(batch_xs_input.shape[0], batch_xs_input.shape[1])  #batch, ch, wid, hei
                batch_xs_target = batch_xs_target.reshape(batch_xs_target.shape[0], batch_xs_target.shape[1]) 


                batch_xs_input, batch_xs_target = torch.from_numpy(batch_xs_input).float().to(device), \
                                                  torch.from_numpy(batch_xs_target).float().to(device)

                assert not torch.isnan(batch_xs_input).any()
                assert not torch.isnan(batch_xs_target).any()

                batch_xs_input_1 = decoder(batch_xs_input)
                batch_xs_input_1 = torch.clamp(batch_xs_input_1, 1e-8, 1 - 1e-8)
                batch_xs_input_1 = batch_xs_input_1.permute((0,2,1)) #to batch, window, ch
                batch_xs_input_1 = batch_xs_input_1.reshape((batch_xs_input_1.shape[0], -1,93))

                y_shifted2 = np.ones((batch_xs_input_1.shape[0],140,93))*0.0 + 0.19  
                y_shifted2 = torch.from_numpy(y_shifted2).float().cuda()

                for ba in range(y_shifted2.shape[0]):
                    y_shifted2[ba,20:20+batch_xs_input_1.shape[1],:] = batch_xs_input_1[ba,:,:]

                batch_xs_input_2 = y_shifted2.reshape((y_shifted2.shape[0], 1, -1))


                output, param2, scale_param = surrogate_model(batch_xs_input_2)
                loss = F.mse_loss(output, batch_xs_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            if epoch % 5 == 0:
            # print cost every epoch
                print("epoch %d: loss %04.6f" % (epoch, loss.item()))
                print("scale_param %04.6f" % (scale_param[0].item()))
                print("shift_param %04.6f" % (param2[0].item()))

        return loss, optimizer, output, batch_xs_target




class PGD(object):

    def __init__(self):
        self.attack_eps = 8 #attack_eps

    def attack1_iter(self, x, y, encoder, decoder, surrogate_model, batch_size=64, attack_steps=25, attack_lr=0.01, random_init=False, target=None, clamp=(0, 1)):
        """
        :param x: Inputs to perturb
        :param y: Corresponding ground-truth labels
        :param net: Network to attack
        :param attack_steps: Number of attack iterations
        :param attack_lr: Learning rate of attacker
        :param random_init: If true, uses random initialization
        :param target: If not None, attacks to the chosen class. Dimension of target should be same as labels
        :return:
        """

        x_adv = x.copy()
        y_adv = y.copy()

        surrogate_model.eval()

        decoder.zero_grad()

        x_adv = torch.FloatTensor(x_adv).to(device)

        optimizer_approx = torch.optim.Adam([x_adv], lr=attack_lr, betas=(0.5, 0.999))



        total_batch = int(x_adv.shape[0] / batch_size)

        x_adv1 = np.ones((x_adv.shape[0], 93, 100)) * 0.5 * 0.0 + 0.19

        tot_mean_loss = 0.0


        for i in range(attack_steps):

            if i % 7 == 0 and i > 0:
                for param_group in optimizer_approx.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

            x_adv.requires_grad = True

            mean_loss = 0.0

            for j in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (j * batch_size) % (x_adv.shape[0])
                batch_xs_input = x_adv[offset:(offset + batch_size), :]
                batch_xs_target = y_adv[offset:(offset + batch_size), :]

                batch_xs_input = batch_xs_input.reshape(batch_xs_input.shape[0], batch_xs_input.shape[1])  #batch, ch, wid, hei
                batch_xs_target = batch_xs_target.reshape(batch_xs_target.shape[0], batch_xs_target.shape[1]) 


                batch_xs_input, batch_xs_target = batch_xs_input, \
                                                  torch.from_numpy(batch_xs_target).float().to(device)


                assert not torch.isnan(batch_xs_input).any()
                assert not torch.isnan(batch_xs_target).any()


                x_adv1a = decoder(batch_xs_input)
                x_adv1a = torch.clamp(x_adv1a, 1e-8, 1 - 1e-8) 

                x_adv1[offset:(offset + batch_size), :] = x_adv1a.detach().cpu().numpy() ##batch, 93, 100


                ################
                batch_xs_input_1 = x_adv1a.permute((0,2,1)) #to batch, 100, 93

                y_shifted2 = np.ones((x_adv1a.shape[0],int(x_adv1a.shape[2]*1.4),93))*0.5 *0.22* 0.0 + 0.19 ###############

                y_shifted2 = torch.from_numpy(y_shifted2).float().cuda()

                for ba in range(y_shifted2.shape[0]):
                    y_shifted2[ba,20:20+batch_xs_input_1.shape[1],:] = batch_xs_input_1[ba,:,:]

                x_adv12 = y_shifted2.reshape((y_shifted2.shape[0], 1, -1)) #batch, 140, 93
                ################


                x_adv2, param2, scale_param = surrogate_model(x_adv12)
                x_adv2 = x_adv2.squeeze(1)

                loss = F.mse_loss(x_adv2, batch_xs_target) #surro out, noisy  
                mean_loss += loss

                optimizer_approx.zero_grad()
                loss.backward()
                optimizer_approx.step()

                
            tot_mean_loss = mean_loss / total_batch

        x_adv = torch.clamp(x_adv, -1, 1)

        mean_loss = tot_mean_loss
        print("mean_loss %04.6f" % (mean_loss.detach().item()))


        return x_adv.detach().cpu().numpy(), x_adv1



