import torch
import numpy as np
import mnist_data
import os
import ae_cnn_shift_scale as vae
import plot_utils
import glob

import argparse
import matplotlib.pyplot as plt
from scipy.io import savemat

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

    #parser.add_argument('--add_noise', type=bool, default=False,
    #                    help='Boolean for adding salt & pepper noise to input image')

    parser.add_argument('--dim_z', type=int, default='20', help='Dimension of latent vector', required=True)

    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP') #500

    parser.add_argument('--learn_rate', type=float, default=1e-4, help='Learning rate for Adam optimizer') #1e-3

    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    parser.add_argument('--PRR', type=bool, default=True,
                        help='Boolean for plot-reproduce-result')

    parser.add_argument('--PRR_n_img_x', type=int, default=30,
                        help='Number of images along x-axis') #PRR_n_img_x 10

    parser.add_argument('--PRR_n_img_y', type=int, default=30,
                        help='Number of images along y-axis') #PRR_n_img_y 10

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
        maxval = 1.0

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
	Input poses is of shape batch_size x channels (3x25x2) x seq_len 
	'''

	batch_size, num_channels, seq_len = poses.size()

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
	range_vec = range_vec
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

    train_total_data, train_size, _, y_train, test_data1, test_labels1, x_test = mnist_data.get_hmd_data(test_id=args.test_id, dataset_dir='HDM_data/')

    n_samples = train_size
    print("n_samples ",n_samples)




    """ create network """
    keep_prob = 0.99
    encoder = vae.Encoder(dim_img, n_hidden, dim_z, keep_prob).to(device)
    decoder = vae.Decoder(dim_z, n_hidden, dim_img, keep_prob).to(device)

    surrogate_model = vae.SurrogateNet2().to(device)
    train_surrogate = vae.SURROGATE()


    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learn_rate)

    enc_scheduler = StepLR(optimizer, step_size=200, gamma=0.5) #30

    """ training """
    # Plot for reproduce performance
    if PRR:
        PRR = plot_utils.Plot_Reproduce_Performance(RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, IMAGE_SIZE_WMNIST,
                                                    IMAGE_SIZE_HMNIST, PRR_resize_factor)

        train_data_ = train_total_data[:, :-mnist_data.NUM_ACTION_LABELS] 

        train_data_label = y_train


        print("train_data_:", train_data_.shape, y_train.shape)
        mean_train_ = np.mean(train_data_, 0)


        test_data11 = test_data1.copy()
        test_labels11 = test_labels1.copy()
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

        x_PRR_o = np.copy(x_PRR)
        x_PRR_shiftscale = np.ones((x_PRR.shape[0],140*93))*0.5
        mean_test = np.mean(x_PRR_o, 0)

        print('performed',mean_train_.shape, mean_test.shape)
        print('performed')


  


        i,j = 0,100
        plt.figure(figsize = (10,5))
        print("x_PRR",x_PRR.shape)


        x_graph = x_PRR
        plt.plot(x_graph[0,:j],'b',label='Original')
        plt.plot(x_graph[1,:j],'b',label='Original')


        if ADD_NOISE == 1 or ADD_NOISE == 5:
            
            ###removal 

            x_removed = np.copy(x_PRR)      #493, 100*93

            x_removed2 = x_removed.reshape((x_removed.shape[0],window,ch))
            x_removed1 = np.transpose(x_removed2, (0, 2, 1)) 
            x_removed = x_removed1.reshape((x_PRR.shape[0], -1))

            missing_param = 20 #10%
            mis_ch = 100 // missing_param
            length_range = int(missing_param*window/100)


            for i in range(x_PRR.shape[0]):

                rnd_ch2 = np.random.randint(31)
                for ci in range(31):
                    if rnd_ch2 == ci:
                        indices = np.random.randint(0,window-length_range,1) 
                        length_removal = np.random.randint(0,length_range,1) 

                        x_removed[i,(ci*3*window)+indices[0]:ci*3*window+(indices[0]+length_removal[0])] = np.zeros((length_removal[0])) + 0.1
                        x_removed[i,((ci*3+1)*window)+indices[0]:(ci*3+1)*window+(indices[0]+length_removal[0])] = np.zeros((length_removal[0])) + 0.1
                        x_removed[i,((ci*3+2)*window)+indices[0]:(ci*3+2)*window+(indices[0]+length_removal[0])] = np.zeros((length_removal[0])) + 0.1

            x_removed21 = x_removed.reshape((x_removed.shape[0],ch,window))
            x_removed3 = np.transpose(x_removed21, (0, 2, 1)) 
            x_removed3 = x_removed3.reshape((x_PRR.shape[0], -1))
            x_PRR = x_removed3


        if ADD_NOISE == 2 or ADD_NOISE == 6:

            ########missing

            x_missing = np.copy(x_PRR) 

            x_missing2 = x_missing.reshape((x_missing.shape[0],window,ch))
            x_missing1 = np.transpose(x_missing2, (0, 2, 1)) 
            x_missing = x_missing1.reshape((x_PRR.shape[0], -1))

            missing_param = 20
            mis_ch = 100 // missing_param
            length_range = int(missing_param*window/100)


            for i in range(x_PRR.shape[0]):
                rnd_ch2 = np.random.randint(31)
                for ci in range(31):
                    #if rnd_ch[ci] == 1:
                    if rnd_ch2 == ci:
                        for x2 in range(missing_param):
                            selected_point = np.random.randint(1,window)
                            x_missing[i,(ci*3*window)+selected_point] = 0.1
                            x_missing[i,((ci*3+1)*window)+selected_point] = 0.1
                            x_missing[i,((ci*3+2)*window)+selected_point] = 0.1

            x_missing21 = x_missing.reshape((x_missing.shape[0],ch,window))
            x_missing3 = np.transpose(x_missing21, (0, 2, 1)) 
            x_missing3 = x_missing3.reshape((x_PRR.shape[0], -1))
            x_PRR = x_missing3

        
        if ADD_NOISE == 3 or ADD_NOISE == 6:
            #####add small value

            x_PRR += np.random.randint(2, size=x_PRR.shape) * 0.2

        if ADD_NOISE == 4 or ADD_NOISE == 5:
            
            #######noise
            noise_param = 20
            x_PRR = x_PRR + np.random.normal(0, noise_param*np.mean(np.std(x_PRR,axis=(1)))/100, x_PRR.shape)

            ###########



        if ADD_NOISE <= 6:
            if SHIFT > 0:

                ###########shift
                tshift_param = 10  #7 #10
                x_shifted = np.copy(x_PRR) 

                ch = 93
                window = 100
                tshift_start = 0
                x_shifted1 = np.copy(x_PRR)
                x_shifted = np.copy(x_PRR) 
                x_shifted = x_shifted.reshape((x_shifted.shape[0],window,ch))
                x_shifted = np.transpose(x_shifted, (0, 2, 1))

                ## Length of signal to move.
                length_range = int(tshift_param*window/100.0)

                print("======== shift length_range",length_range)
                
                x_scale_shifted = np.ones((x_shifted.shape[0],ch,int(window*1.4))) *0.5 * 0.0
                x_scale_shiftedt = np.ones((x_shifted.shape[0],ch,int(window*1.4))) *0.5 * 0.0
                x_scale_shiftedt1 = np.ones((x_shifted.shape[0],int(window*1.4),ch)) *0.5 * 0.0


                time_test = []
                for iterb in range(x_shifted.shape[0]):
                    xid = np.arange(0, int(window*1.4), 1)
                    time_test.append(xid)
                time_test = np.array(time_test)





                for i in range(x_shifted.shape[0]):

                    ######parameters for shift / scale
                    tshift_param2 = 12
                    tscale_param2 = 1.1
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

                    x_scale_shifted[i] = warp(x_scale_shiftedtt, gamma)

                    if i <= 20:
                        print(i,"tscale_param2 %04.2f tshift_param2 %04.2f" % (tscale_param2, tshift_param2))



                x_scale_shifted1a = x_scale_shifted.reshape((x_scale_shifted.shape[0],ch,int(window*1.4)))
                x_scale_shifted2a = np.transpose(x_scale_shifted1a, (0, 2, 1)) 


                x_shifted1 = x_scale_shifted2a.reshape((x_scale_shifted.shape[0],-1))

                x_PRR_shiftscale = x_shifted1
                x_shifted2 = np.ones((x_scale_shifted.shape[0],ch,window))*0.5 * 0.0

                for i in range(x_shifted.shape[0]):
                    for ich in range(ch):
                        x_shifted2[i,ich,:window] = x_scale_shifted[i,ich,20:20+window]

                x_shifted2a = x_shifted2.reshape((x_shifted2.shape[0],ch,window))
                x_shifted2b = np.transpose(x_shifted2a, (0, 2, 1)) 

                x_PRR = x_shifted2b.reshape((x_shifted2b.shape[0],-1))


        if ADD_NOISE == 7:
            ######random noise
            for i in range(x_PRR.shape[0]):
                rnd_noise = np.random.randint(8) #0~7


                ###removal
                if rnd_noise == 1 or rnd_noise == 5:
                    x_removed = np.copy(x_PRR[i])
                    x_removed = x_removed.reshape((window,ch))
                    x_removed = np.transpose(x_removed, (1, 0))


                    missing_param = 20 #5%
                    mis_ch = 100 // missing_param
                    length_range = int(missing_param*window/100)

                    rnd_ch2 = np.random.randint(31) 

                    for ci in range(31):
                        if rnd_ch2 == ci:

                            indices = np.random.randint(0,window-length_range,1) 
                            length_removal = np.random.randint(0,length_range,1) 

                            x_removed[ci*3,indices[0]:indices[0]+length_removal[0]] = np.zeros((length_removal[0])) + 0.1
                            x_removed[(ci*3+1),indices[0]:indices[0]+length_removal[0]] = np.zeros((length_removal[0])) + 0.1
                            x_removed[(ci*3+2),indices[0]:indices[0]+length_removal[0]] = np.zeros((length_removal[0])) + 0.1


                    x_removeda = x_removed.reshape((ch,window))
                    x_removedb = np.transpose(x_removeda, (1, 0)) 

                    x_PRR[i] = x_removedb.reshape(x_PRR[i].shape)

                if rnd_noise == 2 or rnd_noise == 6:

                ########missing

                    x_missing = np.copy(x_PRR[i])
                    x_missing = x_missing.reshape((window,ch))
                    x_missing = np.transpose(x_missing, (1, 0))
            
                    missing_param = 20
                    mis_ch = 100 // missing_param
                    length_range = int(missing_param*window/100)

                    rnd_ch2 = np.random.randint(31) 


                    for ci in range(31):
                        if rnd_ch2 == ci:
                            for x2 in range(missing_param):
                                selected_point = np.random.randint(1,window)
                                x_missing[(ci*3),selected_point] = 0.1
                                x_missing[(ci*3+1),selected_point] = 0.1
                                x_missing[(ci*3+2),selected_point] = 0.1

                    x_missinga = x_missing.reshape((ch,window))
                    x_missingb = np.transpose(x_missinga, (1, 0)) 

                    x_PRR[i] = x_missingb.reshape(x_PRR[i].shape)


                if rnd_noise == 3 or rnd_noise == 6:
                #####add value
                    x_addval = np.copy(x_PRR[i]) 
                    x_addval += np.random.randint(2, size=x_addval.shape) * 0.2
                    x_PRR[i] = x_addval

                if rnd_noise == 4 or rnd_noise == 5:
            
                #######noise
                    x_noise = np.copy(x_PRR[i])
                    noise_param = 20
                    x_noise = x_noise + np.random.normal(0, noise_param*np.mean(np.std(x_noise,axis=(0)))/100, x_noise.shape)
                    x_PRR[i] = x_noise
                ###########

                if SHIFT > 0:
                    ###########shift
                    tshift_param = 10 
                    x_shifted = np.copy(x_PRR[i]) 

                    tshift_start = 0
                    x_shifted1 = np.copy(x_PRR[i])
                    x_shifted = np.copy(x_PRR[i])     
                
                    x_shifted = x_shifted.reshape((1,window,ch))
                    x_shifted = np.transpose(x_shifted, (0, 2, 1))

                    ## Length of signal to move.
                    length_range = int(tshift_param*window/100.0)

               

                    x_scale_shifted = np.ones((1,ch,int(window*1.4))) * 0.0
                    x_scale_shiftedt = np.ones((1,ch,int(window*1.4))) * 0.0
                    x_scale_shiftedt1 = np.ones((1,int(window*1.4),ch)) * 0.0


                    time_test = []
                    for iterb in range(x_shifted.shape[0]):
                        xid = np.arange(0, int(window*1.4), 1)
                        time_test.append(xid)
                    time_test = np.array(time_test)


                    ######parameters for shift / scale
                    tshift_param2 = np.random.randint(30) - 15
                    tscale_param2 = np.random.randint(40) / 100 + 0.8
                    gamma = time_test[0] * tscale_param2 + tshift_param2

                    gamma = gamma.reshape(1,-1)
                    x_shiftedt = x_shifted.reshape(1,ch,window) 

                    gamma[0][0] = 0
                    gamma[0][int(window*1.4)-1] = int(window*1.4)-1
    

                    x_shiftedt = torch.from_numpy(x_shiftedt).float()
                    gamma = torch.from_numpy(gamma).float()

                    gamma = torch.clamp(gamma, min = 0, max = int(window*1.4)-1)
    
                    x_scale_shiftedt[0,:,20:window+20] = x_shiftedt[0,:,:]

                    x_scale_shiftedtt = torch.from_numpy(x_scale_shiftedt[0].reshape(1,ch,-1)).float()

                    x_scale_shifted[0] = warp(x_scale_shiftedtt, gamma)

                    x_scale_shifted1a = x_scale_shifted.reshape((1,ch,int(window*1.4)))
                    x_scale_shifted2a = np.transpose(x_scale_shifted1a, (0, 2, 1))


                    x_shifted1 = x_scale_shifted2a.reshape((x_scale_shifted.shape[0],-1))

                    x_PRR_shiftscale[i] = x_shifted1.copy()
                    x_shifted2 = np.ones((x_scale_shifted.shape[0],ch,window))*0.5 * 0.0

                    for ich in range(ch):
                        x_shifted2[0,ich,:window] = x_scale_shifted[0,ich,20:20+window]

                    x_shifted2a = x_shifted2.reshape((x_shifted2.shape[0],ch,window))
                    x_shifted2b = np.transpose(x_shifted2a, (0, 2, 1))

                    x_PRR[i] = x_shifted2b.reshape((x_shifted2b.shape[0],-1))



                if rnd_noise != 9:
                    x_PRR[i] = x_PRR[i] *0.22+ 0.19



        if ADD_NOISE > 0 or SHIFT > 0:       


            #x_PRR = x_PRR * 0.5 


            print('-----------add noise-------------')  

            x_graph = x_PRR
            plt.plot(x_graph[0,:j],'r',label='Noise Added')
            plt.plot(x_graph[1,:j],'r',label='Noise Added')

            x_graph =  torch.from_numpy(x_PRR.reshape((x_PRR.shape[0],100,93))) #transpose
            x_graph = x_graph.permute((0,2,1)).detach().numpy()

            plt.plot(x_graph[0,0,:j],'y',label='Noise Added')
            plt.plot(x_graph[1,0,:j],'y',label='Noise Added')


        #plt.title('Input')

        if ADD_NOISE == 7:
            x_PRR = torch.from_numpy(x_PRR).float().to(device)

        elif ADD_NOISE != 7:
            x_PRR = torch.from_numpy(x_PRR*0.22+ 0.19).float().to(device)

        if SHIFT >= 0:
            x_PRR_shiftscale = torch.from_numpy(x_PRR_shiftscale*0.22+ 0.19).float().to(device)

    pgd_ = vae.PGD()

    if args.encoder_checkpoint:
        encoder = load_checkpoint(encoder, args.encoder_checkpoint)

    if args.decoder_checkpoint:
        decoder = load_checkpoint(decoder, args.decoder_checkpoint)



    # train
    total_batch = int(n_samples / batch_size)
    min_tot_loss = np.inf

    x_PRRt = np.ones(x_PRR.shape) * 0.5 - np.random.randint(2, size=x_PRR.shape) * 0.2 + np.random.randint(2, size=x_PRR.shape) * 0.3
    x_PRRtest = torch.from_numpy(x_PRRt).float().to(device)

    pgd_result = torch.from_numpy(x_PRRt).float().to(device)


    z_dim = args.dim_z 

    print("zdim",x_PRR.shape[0],z_dim)

    z_approx =  torch.FloatTensor(x_PRR.shape[0], z_dim).uniform_(-1, 1).to(device)

    z_approx = Variable(z_approx)

    optimizer2 = torch.optim.SGD(surrogate_model.parameters(), lr=0.01) #0.01

    pgd_lr= 0.1 
    #####mean estimation###########

    PSNR_mean_base = calc_psnr(Variable(torch.tensor(mean_train_).float()).to(device),Variable(torch.tensor(mean_test).float()).to(device))
    print("PSNR_mean_base :", PSNR_mean_base)

    X_origin = np.zeros(x_PRR_o.shape)

    for i in range(x_PRR_o.shape[0]):
        X_origin[i] += mean_train_

    print("X_origin ",X_origin.shape) #900, 140

    PSNR_mean_tot_base = calc_psnr(Variable(torch.tensor(X_origin).float()).to(device),Variable(torch.tensor(x_PRR_o).float()).to(device))
    print("PSNR_mean_tot_base :", PSNR_mean_tot_base)

    train_data_1 = torch.from_numpy(train_data_).float().to(device) 
    train_data_1 = Variable(train_data_1)



    

    if args.encoder_checkpoint:
        print("===== encoder decoder trainset ===== ")

        encoder.eval()
        decoder.eval()

        n_samples = train_size
        total_batch = int(n_samples / batch_size)

        latent_m_ = np.zeros((batch_size, z_dim)) #128, 64

        latent_m = []

        for i in range(total_batch):
            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (n_samples)
            batch_xs_input = train_data_[offset:(offset + batch_size), :]

            batch_xs_input = batch_xs_input.reshape((batch_xs_input.shape[0],100,93)) #transpose
            batch_xs_input = np.transpose(batch_xs_input, (0, 2, 1))
            batch_xs_input = torch.from_numpy(batch_xs_input).float().to(device)

            mu = encoder(batch_xs_input)

            y = decoder(mu)

            y = torch.clamp(y, 1e-8, 1 - 1e-8)

            latent_train_z = y

            latent_m_ = latent_train_z.detach().cpu().numpy()
            for j in range(latent_m_.shape[0]):
                latent_m.append(latent_m_[j])



        decoded_num = np.array(latent_m)
        print("decoded_num",decoded_num.shape)

        decoded_num = np.transpose(decoded_num, (0, 2, 1))
        x_PRR_save = decoded_num.reshape((decoded_num.shape[0], 100, -1))

        np.save(str(args.results_path)+'_edtrain',x_PRR_save) #original
        np.save(str(args.results_path)+'_edtrain_labels',train_data_label)



    if args.encoder_checkpoint:
        print("===== encoder decoder testset ===== ")

        encoder.eval()
        decoder.eval()

        n_samples = x_test1.shape[0]
        total_batch = int(n_samples / batch_size)

        latent_m_ = np.zeros((batch_size, z_dim)) #128, 64

        latent_m2 = []

        for i in range(total_batch):
            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (n_samples)
            batch_xs_input = x_test1[offset:(offset + batch_size), :]

            batch_xs_input = batch_xs_input.reshape((batch_xs_input.shape[0],100,93)) #transpose
            batch_xs_input = np.transpose(batch_xs_input, (0, 2, 1))
            batch_xs_input = torch.from_numpy(batch_xs_input).float().to(device)

            mu = encoder(batch_xs_input)

            y = decoder(mu)

            y = torch.clamp(y, 1e-8, 1 - 1e-8)

            latent_train_z = y

            latent_m_ = latent_train_z.detach().cpu().numpy()
            for j in range(latent_m_.shape[0]):
                latent_m2.append(latent_m_[j])



        decoded_num = np.array(latent_m2)
        print("decoded_num",decoded_num.shape)
        decoded_num = np.transpose(decoded_num, (0, 2, 1))
        x_PRR_save = decoded_num.reshape((decoded_num.shape[0], 100, -1))

        np.save(str(args.results_path)+'_edtest',x_PRR_save) #original
        np.save(str(args.results_path)+'_edtest_labels',test_labels)



        print("train_data_",train_data_.shape)
        train_data_1 = train_data_.reshape((train_data_.shape[0], 100, -1))
        np.save(str(args.results_path)+'_train',train_data_1) # original
        np.save(str(args.results_path)+'_train_labels',train_data_label)

        x_test11 = x_test1.reshape((x_test1.shape[0], 100, -1))
        np.save(str(args.results_path)+'_test',x_test11) # original
        np.save(str(args.results_path)+'_test_labels',test_labels)

        mdic = {"gt_actions": x_test11, "reconstructed_actions": x_PRR_save}
        savemat("gt_reconac.mat", mdic)

        Xt1_recon = x_PRR_save.reshape((x_PRR_save.shape[0], 100, -1))
        Xt1_org = x_PRR_o.reshape((x_PRR_o.shape[0], 100, -1))[:x_PRR_save.shape[0]]

        PSNR_v = calc_psnr(Variable(torch.tensor(Xt1_recon).float()).to(device),Variable(torch.tensor(Xt1_org).float()).to(device))
        RMSE_v = calc_rmse(Variable(torch.tensor(Xt1_recon).float()).to(device),Variable(torch.tensor(Xt1_org).float()).to(device))
    
        print("PSNR_ED :", PSNR_v)
        print("RMSE_ED :", RMSE_v)

    if args.encoder_checkpoint:
        print("===== encoder test ===== ")

        encoder.eval()

        n_samples = train_size
        total_batch = int(n_samples / batch_size)

        latent_m_ = np.zeros((batch_size, z_dim)) #128, 64

        for i in range(total_batch):
            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (n_samples)
            batch_xs_input = train_data_[offset:(offset + batch_size), :]

            batch_xs_input = batch_xs_input.reshape((batch_xs_input.shape[0],100,93)) #transpose
            batch_xs_input = np.transpose(batch_xs_input, (0, 2, 1))
            batch_xs_input = torch.from_numpy(batch_xs_input).float().to(device)

            mu = encoder(batch_xs_input)

            latent_train_z = mu 

            latent_m_ += latent_train_z.detach().cpu().numpy()

        avg_latent = np.mean(latent_m_, 0)

        decoder.eval()
        avg_latentt = torch.FloatTensor(avg_latent).unsqueeze(0).float().to(device)

        x_avg_latent = decoder(avg_latentt)  #[1, 64]
        x_avg_latent = torch.clamp(x_avg_latent, 1e-8, 1 - 1e-8)

        X_avg = x_avg_latent.reshape(x_avg_latent.shape[0],ch,window)
        X_avg = X_avg.permute((0,2,1))

        Y_avg = mean_test.reshape(x_avg_latent.shape[0], window, ch)

        PSNR_mean_enbase = calc_psnr(Variable(torch.tensor(X_avg).float()),Variable(torch.tensor(Y_avg).float()).to(device))
        print("PSNR_mean_enbase :", PSNR_mean_enbase)

        X_avg_cpu = X_avg.detach().cpu().numpy()

        X_origin = np.zeros(x_PRR_o.shape)
        X_avg_cpu = X_avg_cpu.reshape(1,ch * window)

        for i in range(x_PRR_o.shape[0]):
            X_origin[i] += X_avg_cpu[0]

        PSNR_mean_tot_enbase = calc_psnr(Variable(torch.tensor(X_origin).float()).to(device),Variable(torch.tensor(x_PRR_o).float()).to(device))
        print("PSNR_mean_tot_enbase :", PSNR_mean_tot_enbase)



    best_acc = 0.0
    for epoch in range(n_epochs):

        psnr_t = 0

        encoder.eval()
        decoder.eval()

        if epoch >= 0:
            if epoch == 0:


                x_PRR_m1 = x_PRR.reshape(x_PRR.shape[0], window, -1)

                x_PRR_m1 = x_PRR_m1.permute((0,2,1)) 
                          
                z_approx, pgd_result1 = pgd_.attack1_base2(z_approx, x_PRR_m1, encoder, decoder, surrogate_model, attack_steps=1, attack_lr=pgd_lr)


                y_PRR_1t = pgd_result1.reshape(x_PRR_m1.shape[0], ch, -1)
                y_PRR_1t = y_PRR_1t.permute((0,2,1)) 

                y_PRR1 = y_PRR_1t.reshape(y_PRR_1t.shape[0], -1)

                Xt1 = x_PRR_o.reshape(x_PRR.shape[0], window, -1)

                Yt1 = y_PRR_1t.reshape(y_PRR1.shape[0], window, -1)
                Yobj = x_PRR.reshape(y_PRR1.shape[0], window, -1) #Yobj

                PSNR_v1 = calc_psnr(Variable(torch.tensor(Xt1).float()).to(device),Variable(torch.tensor(Yt1).float()).to(device))
                PSNR_v12 = calc_psnr(Variable(torch.tensor(Xt1).float()).to(device),Variable(torch.tensor(Yobj).float()).to(device))
                
                RMSE_v = calc_rmse(Variable(torch.tensor(Xt1).float()).to(device),Variable(torch.tensor(Yobj).float()).to(device))

                print("---------initial-test------------") 
                print("epoch", epoch, "PSNR_v1: ",PSNR_v1.item(), "PSNR_v2: ",PSNR_v12.item(), "RMSE_v: ",RMSE_v.item())  
                print("-------------------------------") 

                Yobj_save = Yobj.reshape((Yobj.shape[0], 100, -1)).detach().cpu().numpy()
                mdic1 = {"reconstructed_actions_optimized_z": Yobj_save}
                savemat(str(args.results_path)+'_obj.mat', mdic1)
                np.save(str(args.results_path)+'_obj',Yobj_save)

            # Plot for reproduce performance
            if PRR:

                x_PRR_t = x_PRR.reshape(x_PRR.shape[0], window, -1)
                x_PRR_t = x_PRR_t.permute((0,2,1)) 

                y_PRR_t = vae.get_ae(encoder, decoder, x_PRR_t)
                y_PRR_t = y_PRR_t.permute((0,2,1)) 
                y_PRR = y_PRR_t.reshape(y_PRR_t.shape[0], -1)


            ###### with surrogate
                                
                batch_size2 = 32

                loss, optimizer2, _, _ = train_surrogate.train2(optimizer2, z_approx[:], x_PRR_shiftscale[:], encoder, decoder, surrogate_model, batch_size2, steps=20,  learn_rate=0.01)

                z_approx, pgd_result = pgd_.attack1(z_approx, x_PRR_shiftscale, encoder, decoder, surrogate_model, attack_steps=25, attack_lr=pgd_lr) #repeat->o_img



                y_PRR_ta = pgd_result.reshape(x_PRR.shape[0], ch, -1)
                y_PRR_ta = y_PRR_ta.permute((0,2,1)) # to window ch

                y_PRR_t = y_PRR_ta.reshape(x_PRR.shape[0], 1, -1)

                y_PRR = y_PRR_t.reshape(y_PRR_t.shape[0], -1)
                

                
            ###### latnet space no surrogate
                              
                """

                #x_PRR_t = x_PRR.reshape(x_PRR.shape[0], window, -1)
                #x_PRR_m = x_PRR_t.permute((0,2,1)) 
                  
                  
                x_PRR_m = x_PRR.reshape(x_PRR.shape[0], ch, -1)

                z_approx, pgd_result = pgd_.attack1_base(z_approx, x_PRR_m, encoder, decoder, surrogate_model, attack_steps=25, attack_lr=pgd_lr)

                y_PRR_t = pgd_result.reshape(x_PRR_m.shape[0], 1, -1)


                y_PRR = y_PRR_t.reshape(y_PRR_t.shape[0], -1)
                
                """                
                       
                               
####################

                save_flag = 0

                if epoch > 0 or epoch % 3 == 0:

                    # inception distance only works on celeb data set (requires 64x64)
                    fid = 0 # default value for mnist

                    Xt = x_PRR_o.reshape(x_PRR_o.shape[0],1,1,x_PRR_o.shape[1]) #num, ch, hei, wid
                    Yt = y_PRR_t.reshape(y_PRR.shape[0],1,1,y_PRR.shape[1])

                    PSNR_v = calc_psnr(Variable(torch.tensor(Xt).float()).to(device),Variable(torch.tensor(Yt).float()).to(device))
                    RMSE_v = calc_rmse(Variable(torch.tensor(Xt).float()).to(device),Variable(torch.tensor(Yt).float()).to(device))

                    if best_acc < PSNR_v.item():
                        best_acc = PSNR_v.item()
                    print("-------------test--------------") 
                    print("epoch", epoch, "PSNR_v: ",PSNR_v.item(), "RMSE_v:",RMSE_v.item(),"PSNR_best: ",best_acc) 
                    print("-------------------------------") 


                    if epoch >= 0: 
                        print("best_acc", best_acc)
                        save_flag = 1


                        surro_out_train = np.ones((train_data_.shape[0], 140, ch))*0.0 + 0.19

                        if args.save_corrupted_train == 1:
                            print("===== surroagte_trainout (corrupted training set) ===== ")

                            surrogate_model.eval()

                            n_samples = train_size
                            total_batch = int(n_samples / batch_size)

                            latent_m_ = np.zeros((batch_size, z_dim)) #128, 64

                            for i in range(total_batch):

                                offset = (i * batch_size) % (n_samples)
                                batch_xs_input = train_data_[offset:(offset + batch_size), :]

                                batch_xs_input = batch_xs_input.reshape(batch_xs_input.shape[0], window, ch) #batch, 100, 93

                                shifted_train = np.ones((batch_xs_input.shape[0],140,93))*0.0 + 0.19

                                for ba in range(batch_xs_input.shape[0]):
                                    shifted_train[ba,20:20+batch_xs_input.shape[1],:] = batch_xs_input[ba,:,:]


                                batch_xs_input_2 = torch.from_numpy(shifted_train).float().to(device)

                                surro_out_result, _, _ = surrogate_model(batch_xs_input_2) #batch, 140, 93 

                                surro_out_result1 = surro_out_result.reshape(surro_out_result.shape[0], 140, ch) #(x.shape[0], 1, -1)

                                surro_out_train[offset:(offset + batch_size), :] = surro_out_result1.detach().cpu().numpy()


                            surro_out_train_save = surro_out_train.reshape((surro_out_train.shape[0], 140, ch)) #all, 100, 93
                            np.save(str(args.results_path)+'_train_surro_out_'+str(epoch),surro_out_train_save)

                            print("--- Train_surro_out_save ---",surro_out_train_save.shape)


                        surro_out_test = np.ones((test_data1.shape[0], 140, ch))*0.0 + 0.19

                        if args.save_corrupted_train == 5:
                            print("===== surroagte_testout (corrupted testing set) ===== ")

                            surrogate_model.eval()

                            n_samples = test_data1.shape[0]
                            
                            total_batch = int(n_samples / batch_size)

                            latent_m_ = np.zeros((batch_size, z_dim)) #128, 64

                            for i in range(total_batch):

                                offset = (i * batch_size) % (n_samples)

                                batch_xs_input = test_data1[offset:(offset + batch_size), :]

                                batch_xs_input = batch_xs_input.reshape(batch_xs_input.shape[0], window, ch) #batch, 100, 93

                                shifted_train = np.ones((batch_xs_input.shape[0],140,93))*0.5*0.0 + 0.19

                                for ba in range(batch_xs_input.shape[0]):
                                    shifted_train[ba,20:20+batch_xs_input.shape[1],:] = batch_xs_input[ba,:,:]


                                batch_xs_input_2 = torch.from_numpy(shifted_train).float().to(device)

                                surro_out_result, _, _ = surrogate_model(batch_xs_input_2) #batch, 140, 93 

                                surro_out_result1 = surro_out_result.reshape(surro_out_result.shape[0], 140, ch)

                                surro_out_test[offset:(offset + batch_size), :,:] = surro_out_result1.detach().cpu().numpy()

                            x_shifted2 = np.ones((surro_out_test.shape[0],window,ch))*0.5 * 0.0

                            for i in range(x_shifted2.shape[0]):
                                for ich in range(ch):
                                    x_shifted2[i,:window,ich] = surro_out_test[i,20:20+window,ich]

                            x_shifted2a = x_shifted2.reshape((x_shifted2.shape[0],window,ch))

                            surro_out_test_save = x_shifted2a.reshape((surro_out_test.shape[0], window, ch)) #all, 100, 93
                            np.save(str(args.results_path)+'_test_surro_out_'+str(epoch),surro_out_test_save)

                            print("--- Test_surro_out_save ---",surro_out_test_save.shape)


                        
################
                

                name= args.results_path + "/PRR_x_epoch_%02d" % (epoch) + ".jpg"


                i,j = 0,100

                plt.figure(figsize = (10,5))

                x_graph = x_PRR_o
                y_graph = y_PRR_t.detach().cpu().numpy()
                z_graph = x_PRR.detach().cpu().numpy() 

                
                x_graph = x_graph.reshape((x_graph.shape[0],100,93))
                y_graph = y_graph.reshape((y_graph.shape[0],100,93))
                z_graph = z_graph.reshape((z_graph.shape[0],100,93))

                x_graph = np.transpose(x_graph, (0, 2, 1))
                y_graph = np.transpose(y_graph, (0, 2, 1))
                z_graph = np.transpose(z_graph, (0, 2, 1))


                plt.plot(x_graph[0,1,:j],'b',label='Original')
                plt.plot(x_graph[11,10,:j],'b',label='Originald')
                plt.plot(x_graph[60,3,:j],'b',label='Original')
                plt.plot(x_graph[61,80,:j],'b',label='Original')

                plt.plot(y_graph[0,1,:j],'r',label='Generated')
                plt.plot(y_graph[11,10,:j],'r',label='Generated')
                plt.plot(y_graph[60,3,:j],'r',label='Generated')
                plt.plot(y_graph[61,80,:j],'r',label='Generated')

                z_graph1 = x_PRR.detach().cpu().numpy()
                z_graph1 = z_graph1.reshape((z_graph1.shape[0],93,100))        

                if epoch >= 0: #n_epochs -1:
                    y_PRR_save1 = x_test1.reshape((y_graph.shape[0], -1))
                    y_PRR_save1 = y_PRR_save1.reshape((y_graph.shape[0], 100, -1))

                    y_graph = np.transpose(y_graph, (0, 2, 1)) #to 100,93
                    y_PRR_save = y_graph.reshape((y_graph.shape[0], 100, -1))

                    print("y_PRR_save",y_PRR_save.shape)

                    if save_flag == 1:
                        print("save recon test set",y_PRR_save.shape)
                        mdic2 = {"reconstructed_actions_optimized_z": y_PRR_save}
                        savemat(str(args.results_path)+'_'+str(epoch)+'.mat', mdic2)
                        np.save(str(args.results_path)+'_'+str(epoch),y_PRR_save)
                        #np.save(str(args.results_path)+'_labels',test_labels)

                name1="result_x_%02d" % (epoch)
                plt.title(name1)
                #plt.savefig(name)


''' 
    surro_out_train = np.ones((train_data_.shape[0], 140, ch))*0.5*0.0 + 0.19

    if args.save_corrupted_train == 1:
        print("===== surroagte_trainout (corrupted training set) ===== ")

        surrogate_model.eval()

        n_samples = train_size
        total_batch = int(n_samples / batch_size)

        latent_m_ = np.zeros((batch_size, z_dim)) #128, 64

        for i in range(total_batch):

            offset = (i * batch_size) % (n_samples)
            batch_xs_input = train_data_[offset:(offset + batch_size), :]

            #batch_xs_input = batch_xs_input.reshape(batch_xs_input.shape[0], 1, batch_xs_input.shape[1]) #################
            batch_xs_input = batch_xs_input.reshape(batch_xs_input.shape[0], window, ch)

            shifted_train = np.ones((batch_xs_input.shape[0],140,93))*0.5*0.0 + 0.19

            for ba in range(batch_xs_input.shape[0]):
                shifted_train[ba,20:20+batch_xs_input.shape[1],:] = batch_xs_input[ba,:,:]


            batch_xs_input_2 = torch.from_numpy(shifted_train).float().to(device)

            surro_out_result, _, _ = surrogate_model(batch_xs_input_2) #batch, 140, 93 

            #surro_out_result = np.asarray(surro_out_result)

            #print("batch_xs_input_2",batch_xs_input_2.shape)
            #print("surro_out_result",surro_out_result.shape)
            surro_out_result1 = surro_out_result.reshape(surro_out_result.shape[0], 140, ch) #(x.shape[0], 1, -1)
            #out_train = np.ones((surro_out_result.shape[0], window, ch))*0.5

            #for ba in range(surro_out_result.shape[0]):
            #    out_train[ba,:,:] = surro_out_result1[ba,20:20+window,:].detach().cpu().numpy()

            surro_out_train[offset:(offset + batch_size), :] = surro_out_result1.detach().cpu().numpy()



        surro_out_train_save = surro_out_train.reshape((surro_out_train.shape[0], 140, ch)) #all, 140, 93
        np.save(str(args.results_path)+'_train_surro_out',surro_out_train_save)

        print("--- Train_surro_out_save ---",surro_out_train_save.shape)

'''     


if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)
