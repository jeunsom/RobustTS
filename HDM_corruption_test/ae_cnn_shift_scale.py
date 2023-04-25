import  torch
from    torch import nn
from    torch.nn import functional as F
from torch.autograd import Variable
import numpy as np


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

        self.in_dim = 64 #32

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

        self.fc1 = nn.Linear(2048, n_output)
        self.fc2 = nn.Linear(2048, n_output)

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

        h = self.net2(x)

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
    y = torch.clamp(y, 1e-8, 1 - 1e-8) #0~1 ############ -1 + 1e-8

    return y



def get_z(encoder, x):

    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    return z



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
	range_vec = range_vec.cuda()
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

        # Overwrite weight init if passed
        self.weight_init = nl_weight_init


        self.conv1 = nn.Conv1d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        #localization-network
        self.localization2 = nn.Sequential(
            nn.Conv1d(93, 32, kernel_size=7),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
            nn.ReLU(True),
            nn.Conv1d(32, 64, kernel_size=5),
            nn.MaxPool1d(2, stride=2),
            nn.ReLU(True)
        )

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

        self.fc_loc_shift = nn.Sequential(
            nn.Linear(64 * 31, 1),
            nn.Tanh()
        )

        self.fc_loc_scale = nn.Sequential(
            nn.Linear(64 * 31, 1), #32
            nn.Tanh()
        )


        # Initialize the weights/bias with identity transformation
        self.fc_loc_scale.apply(self.weight_init)


    def shiftnet2(self, x):
        xs = self.localization2(x) #batch, 10, 31

        xs = xs.view(xs.shape[0], -1)
        xs2 = xs
        param = self.fc_loc_shift(xs)

        param = torch.clamp(param, -1+1e-8, 1 - 1e-8)
        param = param.view(-1, 1)*15.0

        scale_param = self.fc_loc_scale(xs2)
        scale_param = 1 + scale_param.view(-1, 1) * 0.25 # 



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



    def forward(self, x):

        #####
        x1 = x.reshape(x.shape[0], 140, 93) #batch, window, ch
        x = torch.transpose(x1, 2, 1)
        x, param2, scale_param = self.shiftnet2(x) #batch, ch, window

        ##### return to origin
        x2 = x.reshape(x.shape[0], 93, 140) #batch, ch, window
        x = torch.transpose(x2, 2, 1)
        x = x.reshape(x.shape[0], 1, -1) #batch, window, ch
        #####

        y = self.net1(x)

        x = self.net2(y+x)

        x = torch.clamp(x, 1e-8, 1 - 1e-8)

        #####
        return x, param2, scale_param


class SURROGATE(object):
    def __init__(self):
        super(SURROGATE, self).__init__()

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

        total_batch = int(n_samples / batch_size)
        train_data_ = train_data.clone()
        obj_data_ = obj_data.clone()

        for epoch in range(steps): 

            # Loop over all batches
            surrogate_model.train()

            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_data_[offset:(offset + batch_size), :]
                batch_xs_target = obj_data_[offset:(offset + batch_size), :]

                batch_xs_input = batch_xs_input.reshape(batch_xs_input.shape[0], batch_xs_input.shape[1])  #batch, ch, wid, hei
                batch_xs_target = batch_xs_target.reshape(batch_xs_target.shape[0], batch_xs_target.shape[1]) 

                assert not torch.isnan(batch_xs_input).any()
                assert not torch.isnan(batch_xs_target).any()

                batch_xs_input_1 = decoder(batch_xs_input) #batch, ch, window
                batch_xs_input_1 = torch.clamp(batch_xs_input_1, 1e-8, 1 - 1e-8)



                batch_xs_input_1 = batch_xs_input_1.permute((0,2,1)) #batch, window, ch

                batch_xs_input_1 = batch_xs_input_1.reshape((batch_xs_input_1.shape[0], -1,93))

                y_shifted2 = np.ones((batch_xs_input_1.shape[0],140,93))* 0.0 + 0.19  #

                y_shifted2 = torch.from_numpy(y_shifted2).float().cuda()

                for ba in range(y_shifted2.shape[0]):
                    y_shifted2[ba,20:20+batch_xs_input_1.shape[1],:] = batch_xs_input_1[ba,:,:]

                batch_xs_input_2 = y_shifted2.reshape((y_shifted2.shape[0], 1, -1))


                output, param2, scale_param = surrogate_model(batch_xs_input_2) #batch, window, ch
                loss = F.mse_loss(output, batch_xs_target) # #batch, window, ch

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i == 0 and epoch % 5 == 0:
                    print("scale_param %04.6f" % (scale_param[0].item()))
                    print("shift_param %04.6f" % (param2[0].item()))


            if epoch % 5 == 0:
            # print cost every epoch
                print("epoch %d: loss %04.6f" % (epoch, loss.item()))

        return loss, optimizer, output, batch_xs_target



class PGD(object):

    def __init__(self):
        self.attack_eps = 8


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

        x_adv = x.clone()
        y_adv = y.clone()

        surrogate_model.eval()

        decoder.zero_grad()

        if random_init:
            # Flag to use random initialization
            x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2

        optimizer_approx = torch.optim.Adam([x_adv], lr=attack_lr, betas=(0.5, 0.999))

        mean_loss = 0.0

        for i in range(attack_steps):

            if i % 7 == 0 and i > 0:
                for param_group in optimizer_approx.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

            x_adv.requires_grad = True

            x_adv1 = decoder(x_adv) #batch, ch, winodw
            x_adv1 = torch.clamp(x_adv1, 1e-8, 1 - 1e-8) 

            ################
            batch_xs_input_1 = x_adv1.permute((0,2,1))  #batch, window, ch
            y_shifted2 = np.ones((x_adv1.shape[0],int(x_adv1.shape[2]*1.4),93))*0.0 + 0.19 #with margin


            y_shifted2 = torch.from_numpy(y_shifted2).float().cuda()

            for ba in range(y_shifted2.shape[0]):
                y_shifted2[ba,20:20+batch_xs_input_1.shape[1],:] = batch_xs_input_1[ba,:,:] ##with margin for warping

            x_adv12 = y_shifted2.reshape((y_shifted2.shape[0], 1, -1))

            x_adv2, param2, scale_param = surrogate_model(x_adv12) #IN: batch, window, ch


            x_adv2 = x_adv2.squeeze(1)

            loss = F.mse_loss(x_adv2, y_adv) #surro out, noisy   #batch, window, ch
            mean_loss += loss


            optimizer_approx.zero_grad()
            loss.backward() #retain_graph=True
            optimizer_approx.step()

        mean_loss = mean_loss / attack_steps

        print("mean_loss %04.6f" % (mean_loss.detach().item()))


        x_adv = x_adv.detach()
        x_adv = torch.clamp(x_adv, -1, 1)

        x_adv1 = x_adv1.detach()
        x_adv1 = torch.clamp(x_adv1, *clamp)


        return x_adv, x_adv1



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

        x_adv = x.clone()
        y_adv = y.clone()

        decoder.zero_grad()

        if random_init:
            # Flag to use random initialization
            x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2

        optimizer_approx = torch.optim.Adam([x_adv], lr=attack_lr, betas=(0.5, 0.999))

        mean_loss = 0.0

        for i in range(attack_steps):

            if i % 7 == 0 and i > 0:
                for param_group in optimizer_approx.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

            x_adv.requires_grad = True

            x_adv1 = decoder(x_adv).squeeze(1)
            x_adv1 = torch.clamp(x_adv1, 1e-8, 1 - 1e-8) 


            loss = F.mse_loss(x_adv1, y_adv)
            mean_loss += loss

            optimizer_approx.zero_grad()
            loss.backward()
            optimizer_approx.step()


        mean_loss = mean_loss / attack_steps
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

        x_adv = x.clone()
        y_adv = y.clone()

        decoder.zero_grad()

        mean_loss = 0.0

        for i in range(attack_steps):
            
            x_adv1 = decoder(x_adv).squeeze(1)
            x_adv1 = torch.clamp(x_adv1, 1e-8, 1 - 1e-8) 

            loss = F.mse_loss(x_adv1, y_adv)
            mean_loss += loss


        mean_loss = mean_loss / attack_steps
        print("mean_loss %04.6f" % (mean_loss.detach().item()))


        x_adv = x_adv.detach()
        x_adv = torch.clamp(x_adv, -1, 1)

        x_adv1 = x_adv1.detach()
        x_adv1 = torch.clamp(x_adv1, *clamp)


        return x_adv, x_adv1


