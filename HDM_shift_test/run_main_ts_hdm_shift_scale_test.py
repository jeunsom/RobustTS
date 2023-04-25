import torch
import numpy as np
import data_load
import os
import ae_cnn as vae #load network
import glob

import argparse

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

IMAGE_SIZE_WMNIST = 100*93
IMAGE_SIZE_HMNIST = 1


def load_checkpoint(model, checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn student
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn student with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path)
	model.load_state_dict(model_ckp['model_state_dict'])
	return model


"""parsing and configuration"""


def parse_args():
    desc = "Pytorch implementation of 'Variational AutoEncoder (VAE)'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--results_path', type=str, default='results',
                        help='File path of output images')

    parser.add_argument('--dim_z', type=int, default='20', help='Dimension of latent vector', required=True)

    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP') #500

    parser.add_argument('--learn_rate', type=float, default=1e-4, help='Learning rate for Adam optimizer') #1e-3

    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    parser.add_argument('--PRR', type=bool, default=True,
                        help='Boolean for plot-reproduce-result')

    parser.add_argument('--PRR_n_img_x', type=int, default=30,
                        help='Number of images along x-axis')

    parser.add_argument('--PRR_n_img_y', type=int, default=30,
                        help='Number of images along y-axis')

    parser.add_argument('--PRR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR', type=bool, default=False,
                        help='Boolean for plot-manifold-learning-result')

    parser.add_argument('--PMLR_n_img_x', type=int, default=20,
                        help='Number of images along x-axis')

    parser.add_argument('--PMLR_n_img_y', type=int, default=20,
                        help='Number of images along y-axis')

    parser.add_argument('--PMLR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR_z_range', type=float, default=2.0,
                        help='Range for unifomly distributed latent vector')

    parser.add_argument('--PMLR_n_samples', type=int, default=5000,
                        help='Number of samples in order to get distribution of labeled data')

    parser.add_argument('--test_id', type=int, default=0,
                        help='Number for test subject id')

    parser.add_argument('--trial', type=str, default='sample',
                        help='File log')

    parser.add_argument('--save', type=bool, default=False,
                        help='Boolean for save')

    parser.add_argument('--encoder-checkpoint', default='', type=str, help='optinal pretrained checkpoint for encoder')
    parser.add_argument('--decoder-checkpoint', default='', type=str, help='optinal pretrained checkpoint for decoder')

    parser.add_argument('--add_noise', type=int, default=0, help='Add Noise, removal 1, missing 2, addval 3, noise 4, renoise 5, misaddval 6')
    parser.add_argument('--shift', type=int, default=0, help='no shift 0, shift 1')

    parser.add_argument('--save_corrupted_train', type=int, default=0, help='no 0, yes 1')
    return check_args(parser.parse_args())



def check_args(args):
    # --results_path
    try:
        os.mkdir(args.results_path)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.results_path + '/*')
    for f in files:
        os.remove(f)

    # --save_corrupted_train
    try:
        assert args.save_corrupted_train >= 0
    except:
        print('save_corrupted_train must be int type')
        return None


    # --shift
    try:
        assert args.shift >= 0
    except:
        print('shift must be int type')
        return None

    # --add_noise
    try:
        assert args.add_noise >= 0
    except:
        print('add_noise must be int type')
        return None

    # --dim-z
    try:
        assert args.dim_z > 0
    except:
        print('dim_z must be positive integer')
        return None

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --PRR
    try:
        assert args.PRR == True or args.PRR == False
    except:
        print('PRR must be boolean type')
        return None

    if args.PRR == True:
        # --PRR_n_img_x, --PRR_n_img_y
        try:
            assert args.PRR_n_img_x >= 1 and args.PRR_n_img_y >= 1
        except:
            print('PRR : number of images along each axis must be larger than or equal to one')

        # --PRR_resize_factor
        try:
            assert args.PRR_resize_factor > 0
        except:
            print('PRR : resize factor for each displayed image must be positive')

    # --PMLR
    try:
        assert args.PMLR == True or args.PMLR == False
    except:
        print('PMLR must be boolean type')
        return None

    if args.PMLR == True:
        try:
            assert args.dim_z == 2
        except:
            print('PMLR : dim_z must be two')

        # --PMLR_n_img_x, --PMLR_n_img_y
        try:
            assert args.PMLR_n_img_x >= 1 and args.PMLR_n_img_y >= 1
        except:
            print('PMLR : number of images along each axis must be larger than or equal to one')

        # --PMLR_resize_factor
        try:
            assert args.PMLR_resize_factor > 0
        except:
            print('PMLR : resize factor for each displayed image must be positive')

        # --PMLR_z_range
        try:
            assert args.PMLR_z_range > 0
        except:
            print('PMLR : range for unifomly distributed latent vector must be positive')

        # --PMLR_n_samples
        try:
            assert args.PMLR_n_samples > 100
        except:
            print('PMLR : Number of samples in order to get distribution of labeled data must be large enough')

    return args



def save(path, encoder, decoder, optimizer, epoch):
    trial_id = args.trial

    torch.save({
        'epoch': epoch,
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, '{}{}_{}_epoch{}.pth.tar'.format(path, "encoder", trial_id, epoch))
    torch.save({
        'epoch': epoch,
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, '{}{}_{}_epoch{}.pth.tar'.format(path, "decoder", trial_id, epoch))

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        maxval = 1.0 #max value
        return 20 * torch.log10(maxval*1.0 / torch.sqrt(mse))


class RMSE:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "RMSE"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return torch.sqrt(mse)


def warp(poses, gamma): 
	'''
	Warping Function
	'''

	batch_size, num_channels, seq_len = poses.size()

	#poses = poses.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_joints*num_channels)

	poses = poses.permute(0,2,1)
	pose_vec_len = num_channels
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
	range_vec = range_vec #.cuda()  #.cuda()
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

	output = output.permute(0,2,1)

	return output




def main(args):

    calc_psnr = PSNR()
    calc_rmse = RMSE()

    np.random.seed(222)

    device = torch.device('cuda')

    RESULTS_DIR = args.results_path
    ADD_NOISE = args.add_noise
    n_hidden = args.n_hidden
    window = 100
    ch = 93
    SHIFT = args.shift

    dim_img = window*ch
    dim_z = args.dim_z

    # train
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate

    # Plot
    PRR = args.PRR  # Plot Reproduce Result
    PRR_n_img_x = args.PRR_n_img_x  # number of images along x-axis in a canvas
    PRR_n_img_y = args.PRR_n_img_y  # number of images along y-axis in a canvas
    PRR_resize_factor = args.PRR_resize_factor  # resize factor for each image in a canvas

    PMLR = args.PMLR  # Plot Manifold Learning Result
    PMLR_n_img_x = args.PMLR_n_img_x  # number of images along x-axis in a canvas
    PMLR_n_img_y = args.PMLR_n_img_y  # number of images along y-axis in a canvas
    PMLR_resize_factor = args.PMLR_resize_factor  # resize factor for each image in a canvas
    PMLR_z_range = args.PMLR_z_range  # range for random latent vector
    PMLR_n_samples = args.PMLR_n_samples  # number of labeled samples to plot a map from input data space to the latent space

    """ prepare data """
    train_total_data, train_size, _, y_train, test_data1, test_labels1, x_test = data_load.get_hmd_data(test_id=args.test_id, dataset_dir='HDM_data/')
    n_samples = train_size
    print("n_samples ",n_samples)


    """ create network """
    keep_prob = 0.99
    encoder = vae.Encoder(dim_img, n_hidden, dim_z, keep_prob).to(device)
    decoder = vae.Decoder(dim_z, n_hidden, dim_img, keep_prob).to(device)

    surrogate_model = vae.SurrogateNet2().to(device)
    train_surrogate = vae.SURROGATE()



    """ prepare data """

    if PRR:

        train_data_ = train_total_data[:, :-data_load.NUM_ACTION_LABELS] # 497, 100*93
        train_data_label = y_train


        print("train_data_:", train_data_.shape, y_train.shape)
        mean_train_ = np.mean(train_data_, 0)


        x_test11 = x_test.copy()


        s = np.arange(test_data1.shape[0])
        np.random.shuffle(s)
        test_data2 = test_data1[s,:]
        test_labels2 = test_labels1[s,:]
        x_test12 = x_test[s,:]



        test_data = test_data2[:, :]
        test_labels = test_labels2[:, :]
        x_test1 = x_test12[:, :].copy()


        print('-------------Test data #-----------------')
        lablecount = test_labels.sum(axis=0)
        print("lablecount",lablecount)

        x_PRR = test_data[:, :].copy() #497

        x_PRR_o = np.copy(x_PRR) #original
        x_PRR_shiftscale = np.ones((x_PRR.shape[0],140*93))*0.5 #temporal


 
        """ add noise to data """

        if ADD_NOISE <= 6:
            if SHIFT > 0:

                ###########shift nosie##########

                x_shifted = np.copy(x_PRR) 

                ch = 93
                window = 100
                tshift_start = 0
                #x_shifted1 = np.copy(x_PRR)
                x_shifted = np.copy(x_PRR)      #493, 100*93

                x_shifted = x_shifted.reshape((x_shifted.shape[0],window,ch))
                x_shifted = np.transpose(x_shifted, (0, 2, 1)) 


                
                x_scale_shifted = np.zeros((x_shifted.shape[0],ch,int(window*1.4)))
                x_scale_shiftedt = np.zeros((x_shifted.shape[0],ch,int(window*1.4)))
                x_scale_shiftedt1 = np.zeros((x_shifted.shape[0],int(window*1.4),ch))

                time_test = []
                for iterb in range(x_shifted.shape[0]):
                    xid = np.arange(0, int(window*1.4), 1)
                    time_test.append(xid)
                time_test = np.array(time_test)


                for i in range(x_shifted.shape[0]):

                    ######parameters for shift / scale
                    tshift_param2 = 12 #time shift
                    tscale_param2 = 1.1 #time scale
                    gamma = time_test[i] * tscale_param2 + tshift_param2

                    gamma = gamma.reshape(1,-1)
                    x_shiftedt = x_shifted[i].reshape(1,ch,window) 

                    gamma[0][0] = 0
                    gamma[0][int(window*1.4)-1] = int(window*1.4)-1


                    x_shiftedt = torch.from_numpy(x_shiftedt).float()
                    gamma = torch.from_numpy(gamma).float()

                    gamma = torch.clamp(gamma, min = 0, max = int(window*1.4)-1)

                    x_scale_shiftedt[i,:,20:window+20] = x_shiftedt[0,:,:]

                    x_scale_shiftedtt = torch.from_numpy(x_scale_shiftedt[i].reshape(1,ch,-1)).float()

                    x_scale_shifted[i] = warp(x_scale_shiftedtt, gamma) #make time shift/scale noise

                    if i <= 10:
                        print(i,"tscale_param2 %04.2f tshift_param2 %04.2f" % (tscale_param2, tshift_param2))



                x_scale_shifted1a = x_scale_shifted.reshape((x_scale_shifted.shape[0],ch,int(window*1.4)))
                x_scale_shifted2a = np.transpose(x_scale_shifted1a, (0, 2, 1)) 


                x_shifted1 = x_scale_shifted2a.reshape((x_scale_shifted.shape[0],-1))

                x_PRR_shiftscale = x_shifted1 #noisy/observed data


        if SHIFT >= 0:
            x_PRR_shiftscale = torch.from_numpy(x_PRR_shiftscale*0.22+ 0.19).float().to(device) #add calbration noise


    #load pgd class
    pgd_ = vae.PGD()

    # set encoder/decoder
    if args.encoder_checkpoint:
        encoder = load_checkpoint(encoder, args.encoder_checkpoint)

    if args.decoder_checkpoint:
        decoder = load_checkpoint(decoder, args.decoder_checkpoint)

    # set temporal variable
    x_PRRt = np.ones(x_PRR.shape) * 0.5 - np.random.randint(2, size=x_PRR.shape) * 0.2 + np.random.randint(2, size=x_PRR.shape) * 0.3
    pgd_result = torch.from_numpy(x_PRRt).float().to(device)

    #latent space dimension
    z_dim = args.dim_z #64

    print("zdim",x_PRR.shape[0],z_dim) 


    z_approx =  torch.FloatTensor(x_PRR.shape[0], z_dim).uniform_(-1, 1).to(device)
    z_approx = Variable(z_approx)

    optimizer2 = torch.optim.SGD(surrogate_model.parameters(), lr=0.01)
    pgd_lr= 0.1
    best_acc = 0.0
    for epoch in range(n_epochs):

        encoder.eval()
        decoder.eval()

        if epoch >= 0:
            if PRR:
                
                """ recovery with training surrogate + pgd optimization """
                                
                batch_size2 = 32 #batch size for training surrogate network
               

                #training surrogate net
                if epoch < 1:
                    loss, optimizer2, aa, bb = train_surrogate.train(optimizer2, z_approx[:], x_PRR_shiftscale[:], encoder, decoder, surrogate_model, batch_size2, steps=20,  learn_rate=0.01) #input: optimizer, latent space, observed/noisy data, encoder, decoder, surrogate network, batch_size, iteration, lr
                else:
                    loss, optimizer2, aa, bb = train_surrogate.train(optimizer2, z_approx[:], x_PRR_shiftscale[:], encoder, decoder, surrogate_model, batch_size2, steps=20,  learn_rate=0.01)


                #PGD optimization
                z_approx, pgd_result = pgd_.pgd_opt(z_approx, x_PRR_shiftscale, encoder, decoder, surrogate_model, attack_steps=25, attack_lr=pgd_lr)
                #input: latent space, observed/noisy data, encoder, decoder, surrogate network, iteration, lr


                #set data shape for measuring PSNR
                y_PRR_ta = pgd_result.reshape(x_PRR.shape[0], ch, -1)
                y_PRR_ta = y_PRR_ta.permute((0,2,1))
                y_PRR_t = y_PRR_ta.reshape(x_PRR.shape[0], 1, -1)

                y_PRR = y_PRR_t.reshape(y_PRR_t.shape[0], -1)
                

                if epoch > 0 or epoch % 3 == 0:

                    Xt = x_PRR_o.reshape(x_PRR_o.shape[0],1,1,x_PRR_o.shape[1]) #num, ch, hei, wid
                    Yt = y_PRR_t.reshape(y_PRR.shape[0],1,1,y_PRR.shape[1])

                    PSNR_v = calc_psnr(Variable(torch.tensor(Xt).float()).to(device),Variable(torch.tensor(Yt).float()).to(device))
                    RMSE_v = calc_rmse(Variable(torch.tensor(Xt).float()).to(device),Variable(torch.tensor(Yt).float()).to(device))

                    if best_acc < PSNR_v.item():
                        best_acc = PSNR_v.item()

                    print("-------------test--------------") 
                    print("epoch", epoch, "PSNR_v: ",PSNR_v.item(), "RMSE_v:",RMSE_v.item(),"PSNR_best: ",best_acc) 
                    print("-------------------------------") 




if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)
