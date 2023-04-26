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

        #self.fc1 = nn.Linear(n_hidden, n_output) # h_dim=1024 = 4x256, z_dim=32
        #self.fc2 = nn.Linear(n_hidden, n_output)
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

        #print("encoder x : ",x.shape) #128,3998
        #mu_sigma = self.net(x) #x = 128,784  #128,1500

        # The mean parameter is unconstrained
        #mean = mu_sigma[:, :self.n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        #stddev = 1e-6 + F.softplus(mu_sigma[:, self.n_output:])

        #print("x ",x.shape)
        #x = self.conv1(x)
        #print("x2 ",x.shape)
        #x = self.conv2(x)
        #print("x3 ",x.shape)
        #x = self.conv3(x)
        #print("x4 ",x.shape)
        #x = self.conv4(x)
        #print("x5 ",x.shape)
        #h = x.view(x.size(0), -1)
        #print("h ",h.shape)


        #h = self.net1(x) #mu_sigma
        h = self.net2(x) #mu_sigma

        #mu, logvar = self.fc1(h), self.fc2(h)
        h = self.fc1(h)

        #z = self.fc3(z) #mu_sigma

        #std = logvar.mul(0.5).exp_() 

        #print("h mu",h.shape,mu.shape) #1792, #20
        #return mean, stddev
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
        #print("dec h ", h.shape) #20
        h = self.fc3(h)



        #print(" dec  ")
        #h = h.view(h.size(0), 1024, 1)
        #print("h ",h.shape)
        #x = self.dec1(h)
        #print("x2 ",x.shape)
        #x = self.dec2(x)
        #print("x3 ",x.shape)
        #x = self.dec3(x)
        #print("x4 ",x.shape)
        #x = self.dec4(x)
        #print("x5 ",x.shape)

        #result = self.dec5(x)
        #print("x6 ",result.shape)



        result = self.net1(h)

        return result #* 30





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
    # sampling by re-parameterization technique
    #z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(a)
    y = torch.clamp(y, 1e-8, 1 - 1e-8) #0~1 ############ -1 + 1e-8
    #y = torch.clamp(y, -15, 15) 

    #y = torch.clamp(y, 1e-8, 5 - 1e-8)
    #y = y *5 - 0.5 #torch.clamp(y, -0.5, 4.5) 

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

    #print("x : ",x.shape) #(batchsz(128), img(784))

    # encoding
    h = encoder(x)

    # decoding
    y = decoder(h)
    y = torch.clamp(y, 1e-8, 1 - 1e-8) #1e-8, 1 - 1e-8

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

    #print("x : ",x.shape) #(batchsz(128), img(784))

    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    y = torch.clamp(y, 1e-8, 1 - 1e-8) #1e-8, 1 - 1e-8
#    y = torch.clamp(y, 1e-8, 4 - 1e-8) #0~1


    #y = torch.clamp(y, 1e-8, 5 - 1e-8)

    #y = y *5 - 0.5 #torch.clamp(y, -0.5, 4.5) 
  
    # loss
    # marginal_likelihood2 = torch.sum(x_target * torch.log(y) + (1 - x_target) * torch.log(1 - y)) / batchsz
    #marginal_likelihood = -F.binary_cross_entropy(y, x_target, reduction='sum') / batchsz

    marginal_likelihood = -nn.MSELoss(reduction='sum')(y, x_target) / batchsz
    # print(marginal_likelihood2.item(), marginal_likelihood.item())

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

    #print("x : ",x.shape) #(batchsz(128), img(784))


    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    y = torch.clamp(y, 1e-8, 1 - 1e-8) #1e-8, 1 - 1e-8
#    y = torch.clamp(y, 1e-8, 4 - 1e-8) #0~1

    # encoding
    mu2, sigma2 = encoder(y)
    # sampling by re-parameterization technique
    z2 = mu2 + sigma2 * torch.randn_like(mu2)


    #y = torch.clamp(y, 1e-8, 5 - 1e-8)

    #y = y *5 - 0.5 #torch.clamp(y, -0.5, 4.5) 
  
    # loss
    # marginal_likelihood2 = torch.sum(x_target * torch.log(y) + (1 - x_target) * torch.log(1 - y)) / batchsz
#    marginal_likelihood = -F.binary_cross_entropy(y, x_target, reduction='sum') / batchsz

    #marginal_likelihood = -nn.SmoothL1Loss(reduction='sum')(y, x_target) / batchsz
    marginal_likelihood = -nn.MSELoss(reduction='sum')(y, x_target) / batchsz


    # print(marginal_likelihood2.item(), marginal_likelihood.item())

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



#def imq_kernel(X: torch.Tensor,
#               Y: torch.Tensor,
#               h_dim: int, b_size: int):

def imq_kernel(X,Y,h_dim, b_size):

    batch_size = b_size #X.size(0)

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

    #print("x : ",x.shape) #(batchsz(128), img(784))


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

        #print("z_fake : ",z_fake.shape)
        #print("z_real : ",z_real.shape)

        mmd_loss = imq_kernel(z_real, z_fake, h_dim=encoder.n_output, b_size=batchsz) #torch.Tensor(
        mmd_loss = mmd_loss / batch_size

        total_loss = recon_loss + mmd_loss



    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)



    # encoding2
    #mu2, sigma2 = encoder(y)
    # sampling by re-parameterization technique
    #z2 = mu2 + sigma2 * torch.randn_like(mu2)


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

    #print("x : ",x.shape) #(batchsz(128), img(784))


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


        #print("z_fake : ",z_fake.shape)
        #print("z_real : ",z_real.shape)

        mmd_loss = imq_kernel(z_real, z_fake, h_dim=encoder.n_output, b_size=batchsz) #torch.Tensor(
        mmd_loss = mmd_loss / batch_size

        mmd_loss2 = imq_kernel(z_realt, z_fake, h_dim=encoder.n_output, b_size=batchsz) #torch.Tensor(
        mmd_loss2 = mmd_loss2 / batch_size

        mmd_losst = 0.5*mmd_loss + 0.5*mmd_loss2

        total_loss = recon_loss + mmd_losst



    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)



    # encoding2
    #mu2, sigma2 = encoder(y)
    # sampling by re-parameterization technique
    #z2 = mu2 + sigma2 * torch.randn_like(mu2)


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

	#wa = wa.cuda()
	#wb = wb.cuda()

	output = wa*Ia + wb*Ib

	#output = output.contiguous().view(batch_size, seq_len, num_channels, num_joints)
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
        #self.conv2_drop = nn.Dropout1d()
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
            #nn.Conv1d(16, 16, kernel_size=5, stride=2), #4
            #nn.ReLU(),
            #nn.Conv1d(8, 1, kernel_size=5, stride=1),  #256 n_output #h_dim=1024 #2
            #nn.Sigmoid(),
            #nn.ConvTranspose1d(16, 16, kernel_size=6, stride=2),
            #nn.ReLU(),
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=1), #6, #2
            nn.Tanh(), #ReLU #Tanh() 22.066
            nn.ConvTranspose1d(8, 1, kernel_size=5, stride=1), #6
            nn.Tanh(), 
            nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0),
            #nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0), #, bias = True
        )



        self.net2 = nn.Sequential(
            nn.Sigmoid(),
        )

        self.fc_net1 = nn.Sequential(
            #nn.Linear(10 * 3 * 3, 32),
            nn.Linear(10 * 996, 512),
            #nn.ReLU(True),
            nl, 
            nn.Linear(512, 64),
            #nn.ReLU(True),
            nl, 
            nn.Linear(64, 512),
            nl,
            #nn.ReLU(True),
            nn.Linear(512, 100 * 40)
            #nl,
            #nn.Linear(10 * 996, 40*100)
        )

        self.fc_net1a = nn.Sequential(
            #nn.Linear(10 * 3 * 3, 32),
            nn.Linear(10 * 996, 1024),
            #nn.ReLU(True),
            nl, 
            nn.Linear(1024, 512),
            #nn.ReLU(True),
            nl, 
            nn.Linear(512, 1024),
            nl,
            #nn.ReLU(True),
            nn.Linear(1024, 10 * 400)
            #nl,
            #nn.Linear(10 * 996, 40*100)
        )

        self.fc_net_a = nn.Sequential( #22.3
            #nn.Linear(10 * 3 * 3, 32),
            nn.Linear(10 * 2321, 10*728),
            #nn.ReLU(True),
            nl, 
            nn.Linear(10*728, 10*512),
            #nn.ReLU(True),
            nl, 
            #nn.ReLU(True),
            nn.Linear(10*512, 100 * 93)
            #nl,
            #nn.Linear(10 * 996, 40*100)
        )

        self.fc_net_b = nn.Sequential( #22.3
            #nn.Linear(10 * 3 * 3, 32),
            nn.Linear(10 * 2321, 10*512),
            #nn.ReLU(True),
            nl, 
            nn.Linear(10*512, 10*128),
            #nn.ReLU(True),
            nl, 
            #nn.ReLU(True),
            nn.Linear(10*128, 100 * 93)
            #nl,
            #nn.Linear(10 * 996, 40*100)
        )

        self.fc_net = nn.Sequential( #23.9
            #nn.Linear(10 * 3 * 3, 32),
            nn.Linear(10 * 3251, 10*512),
            #nn.ReLU(True),
            nl, 
            nn.Linear(10*512, 10*96), #10*96
            #nn.ReLU(True),
            nl, 
            #nn.ReLU(True),
            nn.Linear(10*96, 140 * 93)
            #nl,
            #nn.Linear(10 * 996, 40*100)
        )

        self.fc_netaa = nn.Sequential( #21.8
            #nn.Linear(10 * 3 * 3, 32),
            nn.Linear(10 * 2321, 10*1024),
            #nn.ReLU(True),
            nl, 
            nn.Linear(10*1024, 10*1024),
            #nn.ReLU(True),
            nl, 
            #nn.ReLU(True),
            nn.Linear(10*1024, 100 * 93)
            #nl,
            #nn.Linear(10 * 996, 40*100)
        )


        self.fc_net.apply(self.weight_init)
        self.fc_net[0].apply(first_layer_init)

        self.fc_net2 = nn.Sequential(
            #nn.Linear(10 * 3 * 3, 32),
            nn.Linear(140, 31 * 10),
            nl,
            nn.Linear(10 * 31, 64),
            #nn.ReLU(True),
            nl, 
            nn.Linear(64, 31 * 10),
            nl,
            #nn.ReLU(True),
            nn.Linear(31*10, 140)
        )

        self.fc_net2.apply(self.weight_init)
        self.fc_net2[0].apply(first_layer_init)


        self.fc_loc_shift = nn.Sequential(
            #nn.Linear(10 * 3 * 3, 32),
            nn.Linear(64 * 31, 1), #10 * 2321 #10*21
            #nn.ReLU(True),
        #    nn.LeakyReLU(0.1),
            #nl,
        #    nn.Linear(32, 16),
            #nl,
            #nn.ReLU(True),
        #    nn.LeakyReLU(0.1), #########################
            #nn.Dropout(p=0.2), #0.2
            #nn.Linear(1024, 1),
        #    nn.Dropout(p=0.1), #0.1 #########
            nn.Tanh()
            #nn.Sigmoid()
        )

        self.fc_loc_scale = nn.Sequential(
            #nn.Linear(10 * 3 * 3, 32),
            nn.Linear(64 * 31, 1), #32
            #nn.ReLU(True),
            #nl,
       #     nn.LeakyReLU(0.1),
       #     nn.Linear(32, 16),
            #nl,
       #     nn.LeakyReLU(0.1), ###################
       #     nn.Linear(16, 1),
            #nn.Dropout(p=0.1), #0.1 ##########
            nn.Tanh()
            #nn.Sigmoid()
        )


        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        #self.fc_loc_shift[0].weight.data.zero_() ##################
        #self.fc_loc_shift[2].weight.data.zero_()
        #self.fc_loc_shift[4].weight.data.zero_()
        #self.fc_loc_shift[4].bias.data.copy_(torch.tensor([1], dtype=torch.float)) ######
    #    self.fc_loc_shift.apply(self.weight_init)
        #self.fc_loc_shift[0].apply(first_layer_init)

        #self.fc_loc_scale[2].weight.data.zero_() ######
    #    self.fc_loc_scale[2].bias.data.copy_(torch.tensor([1], dtype=torch.float)) ######

        #self.fc_loc_scale[4].weight.data.zero_()
        #self.fc_loc_scale[4].bias.data.copy_(torch.tensor([1], dtype=torch.float))

        self.fc_loc_scale.apply(self.weight_init)
    #    self.fc_loc_scale[0].apply(first_layer_init)


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

        #print("xs2",xs.shape)
        xs = xs.view(xs.shape[0], -1) #10x3x3(90) -> 10x31
        xs2 = xs
        param = self.fc_loc_shift(xs)
        #param = torch.exp(param) *10.0

        param = torch.clamp(param, -1+1e-8, 1 - 1e-8)
        #param = torch.clamp(param, 1e-8, 1 - 1e-8)
        param = param.view(-1, 1)*15.0

        scale_param = self.fc_loc_scale(xs2)
        #scale_param = torch.clamp(scale_param, 1e-8, 1 - 1e-8)
   #     scale_param = scale_param.view(-1, 1) * 2.0 #1.25 # 
        scale_param = 1 + scale_param.view(-1, 1) * 0.25 #1.25 # 



        x_test = []
        for iterb in range(x.shape[0]):
            xid = np.arange(0, x.shape[2], 1)
            x_test.append(xid)
        x_test = np.array(x_test)

        #print("scale_param",scale_param.shape)
        x_test = torch.from_numpy(x_test).float().cuda()
        gamma = scale_param*x_test + param #+ param #19.6 #



        #gamma = scale_param*x_test - param 
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
        #print("x:",x.shape) #128, 1, 4000
        #print("xs:",xs.shape)

        xs = xs.view(-1, 10 * 3251) #10x3x3(90) -> 10x31
        xs = self.fc_net(xs)
        #xs = self.fc_net_a(xs)


        #xs = self.fc_net2(x)
        #theta = theta.view(-1, 2, 3)

        #x = x.reshape(x.shape[0], 1, x.shape[2],1) # 1 channel

        #grid = F.affine_grid(theta, x.size()) #4d
        #x = F.grid_sample(x, grid)

        xs = xs.view(-1, 1, 140*93)

        #x = x*xs
        x = x*F.sigmoid(xs) #########

        return x


    def forward(self, x):
        # transform the input
        #print("x1",x.shape) #128, 1, 140

        #x = x.reshape(x.shape[0], 100, 93)
        #x = torch.transpose(x, 2, 1)
        #x, param2, scale_param = self.shiftnet2(x)

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
        #y = self.stn(x) 
   #     y = self.linearnet(x) 

        y = self.net1(x) #128,1,124   #20.5718
        #print("y4",y.shape)

        x = self.net2(y+x)
        #x = y+x

        x = torch.clamp(x, 1e-8, 1 - 1e-8)

        ##x = x.reshape(x.shape[0], 93, 100)


        #####
        #x1 = x.reshape(x.shape[0], 140, 93) #batch, window, ch
        #x = torch.transpose(x1, 2, 1)
        #x, param2, scale_param = self.shiftnet2(x)

        #####
        #x2 = x.reshape(x.shape[0], 93, 140) #batch, ch, window
        #x = torch.transpose(x2, 2, 1)
        #x = x.reshape(x.shape[0], 1, -1)


        #x1 = x.reshape(x.shape[0], 140, 93) #batch, window, ch
        #x1 = torch.transpose(x1, 2, 1)
        #x, param2, scale_param = self.shiftnet2(x1)

        ##x2 = x.reshape(x.shape[0], 93, 140) #batch, window, ch
        ##x = torch.transpose(x2, 2, 1)
        ##x = x.reshape(x.shape[0], 1, -1)

        #####
        return x, param2, scale_param #F.log_softmax(x, dim=1)



class SURROGATE(object):
    def __init__(self):
        super(SURROGATE, self).__init__()

    def train(self, optimizer, train_data, obj_data, encoder, decoder, surrogate_model, batch_size, steps=25, learn_rate=0.01, clamp=(0, 1)):
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


        # train #950, 7, 128 = 896
        total_batch = int(n_samples / batch_size)
        train_data_ = train_data.clone()
        obj_data_ = obj_data.clone()

        #optimizer = torch.optim.SGD(surrogate_model.parameters(), lr=0.01)


        for epoch in range(steps): 

            # Loop over all batches
            surrogate_model.train()

            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_data_[offset:(offset + batch_size), :]
                batch_xs_target = obj_data_[offset:(offset + batch_size), :]

                #print("batch_xs_input",batch_xs_input.shape)

                batch_xs_input = batch_xs_input.reshape(batch_xs_input.shape[0], 1, batch_xs_input.shape[1])  #batch, ch, wid, hei
                batch_xs_target = batch_xs_target.reshape(batch_xs_target.shape[0], 1, batch_xs_target.shape[1]) 


                #batch_xs_input, batch_xs_target = torch.from_numpy(batch_xs_input).float().to(device), \
                #                                  torch.from_numpy(batch_xs_target).float().to(device)

                assert not torch.isnan(batch_xs_input).any()
                assert not torch.isnan(batch_xs_target).any()

                output = surrogate_model(batch_xs_input)
                loss = F.mse_loss(output, batch_xs_target) # #loss = nn.L1Loss()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            if epoch % 5 == 0:
            # print cost every epoch
                print("epoch %d: loss %04.6f" % (epoch, loss.item()))


        return loss, optimizer, output, batch_xs_target

    def train2(self, optimizer, train_data, obj_data, encoder, decoder, surrogate_model, batch_size, steps=25, learn_rate=0.01, clamp=(0, 1)):
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

        # train #950, 7, 128 = 896
        total_batch = int(n_samples / batch_size)


        #train_data_ = torch.from_numpy(train_data).float().to(device).clone()

        train_data_ = train_data.clone()
        obj_data_ = torch.from_numpy(obj_data).float().to(device).clone()

        #optimizer = torch.optim.SGD(surrogate_model.parameters(), lr=0.01)


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


                #batch_xs_input, batch_xs_target = torch.from_numpy(batch_xs_input).float().to(device), \
                #                                  torch.from_numpy(batch_xs_target).float().to(device)

                assert not torch.isnan(batch_xs_input).any()
                assert not torch.isnan(batch_xs_target).any()

                #print("batch_xs_input",batch_xs_input.shape) batch_xs_input.view(batch_xs_input.shape[0], -1)
                batch_xs_input_1 = decoder(batch_xs_input) #.squeeze(1) #x_adv-64 x_adv1-140
                batch_xs_input_1 = torch.clamp(batch_xs_input_1, 1e-8, 1 - 1e-8)


                batch_xs_input_1 = batch_xs_input_1.reshape((batch_xs_input_1.shape[0], -1,93))

                y_shifted2 = np.ones((batch_xs_input_1.shape[0],140,93))*0.5

                #print("y_shifted2",y_shifted2.shape,batch_xs_target.shape)

                y_shifted2 = torch.from_numpy(y_shifted2).float().cuda()

                for ba in range(y_shifted2.shape[0]):
                    y_shifted2[ba,20:20+batch_xs_input_1.shape[1],:] = batch_xs_input_1[ba,:,:]
                    #for ich in range(93):
                    #    y_shifted2[ba,25:25+batch_xs_input_1.shape[1],ich] = batch_xs_input_1[ba,:,ich]

                batch_xs_input_2 = y_shifted2.reshape((y_shifted2.shape[0], 1, -1))


                output, param2, scale_param = surrogate_model(batch_xs_input_2)
                loss = F.mse_loss(output, batch_xs_target) # #loss = nn.L1Loss()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            if epoch % 5 == 0:
            # print cost every epoch
                print("epoch %d: loss %04.6f" % (epoch, loss.item()))
                #print("scale_param %04.6f" % (scale_param.item()))
                #print("shift_param %04.6f" % (param2.item()))
                print("scale_param %04.6f" % (scale_param[0].item()))
                print("shift_param %04.6f" % (param2[0].item()))

        return loss, optimizer, output, batch_xs_target


    def train22(self, optimizer, train_data, obj_data, encoder, decoder, surrogate_model, batch_size, steps=25, learn_rate=0.01, clamp=(0, 1)):
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

        # train #950, 7, 128 = 896
        total_batch = int(n_samples / batch_size)
        train_data_ = train_data.copy()
        obj_data_ = obj_data.copy()

        #optimizer = torch.optim.SGD(surrogate_model.parameters(), lr=0.01)


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

                #print("batch_xs_input",batch_xs_input.shape) batch_xs_input.view(batch_xs_input.shape[0], -1)
                batch_xs_input_1 = decoder(batch_xs_input) #.squeeze(1) #x_adv-64 x_adv1-140
                batch_xs_input_1 = torch.clamp(batch_xs_input_1, 1e-8, 1 - 1e-8)

                batch_xs_input_1 = batch_xs_input_1.permute((0,2,1)) #to batch, window, ch

                batch_xs_input_1 = batch_xs_input_1.reshape((batch_xs_input_1.shape[0], -1,93))

                y_shifted2 = np.ones((batch_xs_input_1.shape[0],140,93))*0.5 *0.22* 0.0 + 0.19  #################

                #print("y_shifted2",y_shifted2.shape,batch_xs_target.shape)

                y_shifted2 = torch.from_numpy(y_shifted2).float().cuda()

                for ba in range(y_shifted2.shape[0]):
                    y_shifted2[ba,20:20+batch_xs_input_1.shape[1],:] = batch_xs_input_1[ba,:,:]
                    #for ich in range(93):
                    #    y_shifted2[ba,25:25+batch_xs_input_1.shape[1],ich] = batch_xs_input_1[ba,:,ich]

                batch_xs_input_2 = y_shifted2.reshape((y_shifted2.shape[0], 1, -1))


                output, param2, scale_param = surrogate_model(batch_xs_input_2)
                loss = F.mse_loss(output, batch_xs_target) # #loss = nn.L1Loss()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            if epoch % 5 == 0:
            # print cost every epoch
                print("epoch %d: loss %04.6f" % (epoch, loss.item()))
                #print("scale_param %04.6f" % (scale_param.item()))
                #print("shift_param %04.6f" % (param2.item()))
                print("scale_param %04.6f" % (scale_param[0].item()))
                print("shift_param %04.6f" % (param2[0].item()))

        return loss, optimizer, output, batch_xs_target




class PGD(object):

    def __init__(self):
        self.attack_eps = 8 #attack_eps

    def attack1(self, x, y, encoder, decoder, surrogate_model, attack_steps=25, attack_lr=0.01, random_init=False, target=None, clamp=(0, 1)):
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

        x_adv = x.clone() #+ 0.1
        y_adv = y.clone() #*0.22 #noise

        surrogate_model.eval()

        decoder.zero_grad()
        #mu, sigma = encoder(y_adv)
        ## sampling by re-parameterization technique
        #z = mu + sigma * torch.randn_like(mu)

        if random_init:
            # Flag to use random initialization
            x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 #* self.attack_eps


        #print("x_adv1",x_adv[0][:5])
        #x_adv.requires_grad = True

        optimizer_approx = torch.optim.Adam([x_adv], lr=attack_lr, betas=(0.5, 0.999)) #, betas=(opt.beta1, 0.999)

        mean_loss = 0.0

        for i in range(attack_steps):
            #print("x_adv1",x_adv[0][:5])

            if i % 7 == 0 and i > 0:
                for param_group in optimizer_approx.param_groups:
                    param_group['lr'] = param_group['lr']*0.5
                #    print("LR", param_group['lr'])



            x_adv.requires_grad = True

            #print("i attack", i, attack_steps)

            x_adv1 = decoder(x_adv) #.squeeze(1) #x_adv-64 x_adv1-140
            x_adv1 = torch.clamp(x_adv1, 1e-8, 1 - 1e-8) 

            ################
            #print("x_adv1",x_adv1.shape) #960, 1, 140

            batch_xs_input_1 = x_adv1.reshape((x_adv1.shape[0], -1,93))
            y_shifted2 = np.ones((x_adv1.shape[0],int(x_adv1.shape[2]*1.4),93))

            #print("y_shifted2",y_shifted2.shape,batch_xs_target.shape)

            y_shifted2 = torch.from_numpy(y_shifted2).float().cuda()

            for ba in range(y_shifted2.shape[0]):
                y_shifted2[ba,20:20+batch_xs_input_1.shape[1],:] = batch_xs_input_1[ba,:,:]
                #for ich in range(93):
                #    y_shifted2[ba,25:25+batch_xs_input_1.shape[1],ich] = batch_xs_input_1[ba,:,ich]

            x_adv12 = y_shifted2.reshape((y_shifted2.shape[0], 1, -1))
            ################

            #x_adv2, param2, scale_param = surrogate_model(x_adv1).squeeze(1)
            x_adv2, param2, scale_param = surrogate_model(x_adv12)


            x_adv2 = x_adv2.squeeze(1)

            #x_adv2.requires_grad = True #.retain_grad()
            #x_adv2.retain_grad()

            
            #print("adv2", x_adv2.shape, y_adv.shape)
            loss = F.mse_loss(x_adv2, y_adv) #surro out, noisy  
            mean_loss += loss

      #      print("loss %04.6f" % (loss.item()))

            #if i % 5 == 0:
            # print cost every epoch
            #    print("epoch %d: loss %04.6f" % (i, loss.item()))

            optimizer_approx.zero_grad()
            loss.backward() #retain_graph=True
            optimizer_approx.step()

       #     grad2 = x_adv2.grad
       #     grad2 = grad2.sign()


       #     x_adv = x_adv - attack_lr * grad2


            #x_adv = x_adv.detach()
            #x_adv = torch.clamp(x_adv, -1, 1)


            #print("x_adv4",x_adv[0][:5])

            # Projection
            #x_adv = x + torch.clamp(x_adv - x, min=-self.attack_eps, max=self.attack_eps)
            #x_adv = x_adv + torch.clamp(x_adv - x, 1e-8, 1 - 1e-8)
     #       x_adv = x_adv.detach()
     #       x_adv = torch.clamp(x_adv, *(-1, 1))


        mean_loss = mean_loss / attack_steps
        #print("mean_loss", mean_loss.detach())
        print("mean_loss %04.6f" % (mean_loss.detach().item()))


        x_adv = x_adv.detach()
        x_adv = torch.clamp(x_adv, -1, 1)

        x_adv1 = x_adv1.detach()
        x_adv1 = torch.clamp(x_adv1, *clamp)


        return x_adv, x_adv1


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

        x_adv = x.copy() #+ 0.1
        y_adv = y.copy() #*0.22 #noise

        surrogate_model.eval()

        decoder.zero_grad()

        x_adv = torch.FloatTensor(x_adv).to(device)

        optimizer_approx = torch.optim.Adam([x_adv], lr=attack_lr, betas=(0.5, 0.999)) #, betas=(opt.beta1, 0.999)



        total_batch = int(x_adv.shape[0] / batch_size)

        x_adv1 = np.ones((x_adv.shape[0], 93, 100)) * 0.5 * 0.0 + 0.19

        tot_mean_loss = 0.0


        for i in range(attack_steps):
            #print("x_adv1",x_adv[0][:5])

            if i % 7 == 0 and i > 0:
                for param_group in optimizer_approx.param_groups:
                    param_group['lr'] = param_group['lr']*0.5
                #    print("LR", param_group['lr'])

            x_adv.requires_grad = True

            mean_loss = 0.0

            for j in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (j * batch_size) % (x_adv.shape[0])
                batch_xs_input = x_adv[offset:(offset + batch_size), :]
                batch_xs_target = y_adv[offset:(offset + batch_size), :]

                batch_xs_input = batch_xs_input.reshape(batch_xs_input.shape[0], batch_xs_input.shape[1])  #batch, ch, wid, hei
                batch_xs_target = batch_xs_target.reshape(batch_xs_target.shape[0], batch_xs_target.shape[1]) 


                #batch_xs_input, batch_xs_target = torch.from_numpy(batch_xs_input).float().to(device), \
                #                                  torch.from_numpy(batch_xs_target).float().to(device)

                batch_xs_input, batch_xs_target = batch_xs_input, \
                                                  torch.from_numpy(batch_xs_target).float().to(device)


                assert not torch.isnan(batch_xs_input).any()
                assert not torch.isnan(batch_xs_target).any()


                x_adv1a = decoder(batch_xs_input) #.squeeze(1) #x_adv-64 x_adv1-140
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
                loss.backward() #retain_graph=True
                optimizer_approx.step()

            #if i % 5 == 0:
            ## print cost every epoch
            #    print("pgd epoch %d: tot_mean_loss %04.6f" % (i, mean_loss))

                
            tot_mean_loss = mean_loss / total_batch

        #x_adv = x_adv.detach()
        x_adv = torch.clamp(x_adv, -1, 1)

        #x_adv1 = x_adv1.detach()
        #x_adv1 = torch.clamp(x_adv1, *clamp)

        mean_loss = tot_mean_loss #mean_loss / total_batch
        print("mean_loss %04.6f" % (mean_loss.detach().item()))


        return x_adv.detach().cpu().numpy(), x_adv1


    def attack1_base(self, x, y, encoder, decoder, surrogate_model, attack_steps=25, attack_lr=0.01, random_init=False, target=None, clamp=(0, 1)):
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

        x_adv = x.clone() #+ 0.1
        y_adv = y.clone() #*0.22 #noise

       # surrogate_model.eval()

        decoder.zero_grad()
        #mu, sigma = encoder(y_adv)
        ## sampling by re-parameterization technique
        #z = mu + sigma * torch.randn_like(mu)

        if random_init:
            # Flag to use random initialization
            x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 #* self.attack_eps


        #print("x_adv1",x_adv[0][:5])
        #x_adv.requires_grad = True

        optimizer_approx = torch.optim.Adam([x_adv], lr=attack_lr, betas=(0.5, 0.999)) #, betas=(opt.beta1, 0.999)

        mean_loss = 0.0

        for i in range(attack_steps):
            #print("x_adv1",x_adv[0][:5])

            if i % 7 == 0 and i > 0:
                for param_group in optimizer_approx.param_groups:
                    param_group['lr'] = param_group['lr']*0.5
                #    print("LR", param_group['lr'])



            x_adv.requires_grad = True

            #print("i attack", i, attack_steps)

            x_adv1 = decoder(x_adv).squeeze(1) #x_adv-64 x_adv1-140
            x_adv1 = torch.clamp(x_adv1, 1e-8, 1 - 1e-8) 


            loss = F.mse_loss(x_adv1, y_adv) #surro out, noisy  
            mean_loss += loss

      #      print("loss %04.6f" % (loss.item()))


            optimizer_approx.zero_grad()
            loss.backward() #retain_graph=True
            optimizer_approx.step()


            # Projection
            #x_adv = x + torch.clamp(x_adv - x, min=-self.attack_eps, max=self.attack_eps)
            #x_adv = x_adv + torch.clamp(x_adv - x, 1e-8, 1 - 1e-8)
     #       x_adv = x_adv.detach()
     #       x_adv = torch.clamp(x_adv, *(-1, 1))


        mean_loss = mean_loss / attack_steps
        #print("mean_loss", mean_loss.detach())
        print("mean_loss %04.6f" % (mean_loss.detach().item()))


        x_adv = x_adv.detach()
        x_adv = torch.clamp(x_adv, -1, 1)

        x_adv1 = x_adv1.detach()
        x_adv1 = torch.clamp(x_adv1, *clamp)


        return x_adv, x_adv1

    def attack1_base2(self, x, y, encoder, decoder, surrogate_model, attack_steps=1, attack_lr=0.01, random_init=False, target=None, clamp=(0, 1)):
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

        x_adv = x.clone() #+ 0.1
        y_adv = y.clone() #*0.22 #noise

       # surrogate_model.eval()

        decoder.zero_grad()

        mean_loss = 0.0

        for i in range(attack_steps):
            
            x_adv1 = decoder(x_adv).squeeze(1) #x_adv-64 x_adv1-140
            x_adv1 = torch.clamp(x_adv1, 1e-8, 1 - 1e-8) 

            loss = F.mse_loss(x_adv1, y_adv) #surro out, noisy  
            mean_loss += loss


        mean_loss = mean_loss / attack_steps
        #print("mean_loss", mean_loss.detach())
        print("mean_loss %04.6f" % (mean_loss.detach().item()))


        x_adv = x_adv.detach()
        x_adv = torch.clamp(x_adv, -1, 1)

        x_adv1 = x_adv1.detach()
        x_adv1 = torch.clamp(x_adv1, *clamp)


        return x_adv, x_adv1

    def attack2(self, x, y, encoder, decoder, surrogate_model, attack_steps=25, attack_lr=0.1, random_init=False, target=None, clamp=(0, 1)):
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

        x_adv = x.clone() #+ 0.1
        y_adv = y.clone() #*0.22 #noise

        if random_init:
            # Flag to use random initialization
            x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 #* self.attack_eps


        print("x_adv1",x_adv[0][:5])

        for i in range(attack_steps):

            #print("x_adv2",x_adv[0][:5])
            x_adv1 = x_adv*0.35 + 0.196 - 1e-3               #surrogate = *0.78-->35 *0.8-->65

            #print("x_adv3",x_adv1[0][:5])


            #x_adv.requires_grad = True

            x_adv1.requires_grad = True



            loss = F.mse_loss(x_adv1, y_adv) #surro out, noisy  

            #print("loss %04.6f" % (loss.item()))

            #if i % 5 == 0:
            # print cost every epoch
            #    print("epoch %d: loss %04.6f" % (i, loss.item()))


            loss.backward(retain_graph=True)

            grad2 = x_adv1.grad
            grad2 = grad2.sign()


            x_adv = x_adv - attack_lr * grad2

            #print("x_adv4",x_adv[0][:5])

            # Projection
            #x_adv = x + torch.clamp(x_adv - x, min=-self.attack_eps, max=self.attack_eps)
            #x_adv = x_adv + torch.clamp(x_adv - x, 1e-8, 1 - 1e-8)
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *clamp)

        return x_adv

    def attack3(self, x, y, encoder, decoder, surrogate_model, attack_steps=25, attack_lr=0.1, random_init=False, target=None, clamp=(1e-8, 1 - 1e-8)):
        """
        #clamp=(0, 1)
        :param x: Inputs to perturb
        :param y: Corresponding ground-truth labels
        :param net: Network to attack
        :param attack_steps: Number of attack iterations
        :param attack_lr: Learning rate of attacker
        :param random_init: If true, uses random initialization
        :param target: If not None, attacks to the chosen class. Dimension of target should be same as labels
        :return:
        """

        x_adv = x.clone() #+ 0.1
        y_adv = y.clone()  #target

        if random_init:
            # Flag to use random initialization
            x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 #* self.attack_eps


        print("x_adv1",x_adv[0][0])

        for i in range(attack_steps):
            #x_adv.requires_grad = True


            surrogate_model.eval()
            #surrogate_model.zero_grad()




            #net.zero_grad()
            #encoder.zero_grad()
            #decoder.zero_grad()


            # encoding
            #mu, sigma = encoder(x_adv)
            # decoding
            #rey = decoder(mu)
            #rey = torch.clamp(y, 1e-8, 1 - 1e-8)

            #print("pgd x_adv",x_adv.shape) #128,1,140
            #ADD noise (surrogate acting)
            x_adv1 = surrogate_model(x_adv).squeeze(1)


            #print("x_adv1 y",x_adv1.shape,y_adv.shape) #x_adv1 y torch.Size([128, 1, 140]) torch.Size([128, 140])



            x_adv1.retain_grad()

            #print("x_adv2",x_adv1[0][0][:5])



            loss = F.mse_loss(x_adv1, y_adv)
            

           # x_adv2.requires_grad = True
           # print("x_adv",x_adv.shape)

#            loss = F.mse_loss(x_adv, y_adv)
           # x_adv.requires_grad = True

            if i % 5 == 0:
            # print cost every epoch
                print("epoch %d: loss %04.6f" % (i, loss.item()))

            #print("loss",loss)
            loss.backward(retain_graph=True)
  
            #loss.detach().requires_grad = True


            #grad = loss.retain_grad() #.detach()
            #print("grad",grad) #None

            #grad2 = loss.grad
            #grad2 = x_adv.grad
            grad2 = x_adv1.grad
            #print("grad2",grad2)
            grad2 = grad2.sign()

            #loss.backward()
            #grad = x_adv.grad.detach()
            #grad = grad.sign() #integer


#            print("x_adv",x_adv)

            x_adv = x_adv - attack_lr * grad2

#            print("attack_lr * grad2",attack_lr * grad2)
#            print("x_adv1",x_adv)

            # Projection
            #x_adv = x + torch.clamp(x_adv - x, min=-self.attack_eps, max=self.attack_eps)
#            x_adv = x + torch.clamp(x_adv - x, 1e-8, 1 - 1e-8)
            x_adv = x_adv.detach()
            x_adv = torch.clamp(x_adv, *clamp)

#        print("x_adv2",x_adv[0][0])

        return x_adv





# # Gaussian MLP as encoder
# def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob):
#     with tf.variable_scope("gaussian_MLP_encoder"):
#         # initializers
#         w_init = tf.contrib.layers.variance_scaling_initializer()
#         b_init = tf.constant_initializer(0.)
#
#         # 1st hidden layer
#         w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
#         b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
#         h0 = tf.matmul(x, w0) + b0
#         h0 = tf.nn.elu(h0)
#         h0 = tf.nn.dropout(h0, keep_prob)
#
#         # 2nd hidden layer
#         w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
#         b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
#         h1 = tf.matmul(h0, w1) + b1
#         h1 = tf.nn.tanh(h1)
#         h1 = tf.nn.dropout(h1, keep_prob)
#
#         # output layer
#         # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
#         wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2], initializer=w_init)
#         bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
#         gaussian_params = tf.matmul(h1, wo) + bo
#
#         # The mean parameter is unconstrained
#         mean = gaussian_params[:, :n_output]
#         # The standard deviation must be positive. Parametrize with a softplus and
#         # add a small epsilon for numerical stability
#         stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])
#
#     return mean, stddev
#
#
# # Bernoulli MLP as decoder
# def bernoulli_MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):
#     with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
#         # initializers
#         w_init = tf.contrib.layers.variance_scaling_initializer()
#         b_init = tf.constant_initializer(0.)
#
#         # 1st hidden layer
#         w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
#         b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
#         h0 = tf.matmul(z, w0) + b0
#         h0 = tf.nn.tanh(h0)
#         h0 = tf.nn.dropout(h0, keep_prob)
#
#         # 2nd hidden layer
#         w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
#         b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
#         h1 = tf.matmul(h0, w1) + b1
#         h1 = tf.nn.elu(h1)
#         h1 = tf.nn.dropout(h1, keep_prob)
#
#         # output layer-mean
#         wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
#         bo = tf.get_variable('bo', [n_output], initializer=b_init)
#         y = tf.sigmoid(tf.matmul(h1, wo) + bo)
#
#     return y
#
#
# # Gateway
# def autoencoder2(x_hat, x, dim_img, dim_z, n_hidden, keep_prob):
#     # encoding
#     mu, sigma = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob)
#
#     # sampling by re-parameterization technique
#     z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
#
#     # decoding
#     y = bernoulli_MLP_decoder(z, n_hidden, dim_img, keep_prob)
#     y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)
#
#     # loss
#     marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
#     KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)
#
#     marginal_likelihood = tf.reduce_mean(marginal_likelihood)
#     KL_divergence = tf.reduce_mean(KL_divergence)
#
#     ELBO = marginal_likelihood - KL_divergence
#
#     loss = -ELBO
#
#     return y, z, loss, -marginal_likelihood, KL_divergence
#
#
# def decoder(z, dim_img, n_hidden):
#     y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)
#
#     return y
