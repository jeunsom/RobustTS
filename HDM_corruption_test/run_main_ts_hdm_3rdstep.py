import torch
import numpy as np
import mnist_data
import os
import ae_cnn_shift_scale as vae 
import plot_utils
import glob

import argparse
import matplotlib.pyplot as plt

from fid_score2 import get_activations, calculate_frechet_distance
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from scipy.io import savemat

IMAGE_SIZE_WMNIST = 100*93
IMAGE_SIZE_HMNIST = 1

"""parsing and configuration"""


def parse_args():
    desc = "Pytorch implementation of 'Variational AutoEncoder (VAE)'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--results_path', type=str, default='results',
                        help='File path of output images')

    #parser.add_argument('--add_noise', type=bool, default=False,
    #                    help='Boolean for adding salt & pepper noise to input image')

    parser.add_argument('--dim_z', type=int, default='20', help='Dimension of latent vector', required=True)

    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP')

    parser.add_argument('--learn_rate', type=float, default=1e-4, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

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

    parser.add_argument('--add_noise', type=int, default=0, help='Add Noise, removal 1, missing 2, addval 3, noise 4, renoise 5, misaddval 6')


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




def main(args):

    calc_psnr = PSNR()

    np.random.seed(222)


    device = torch.device('cuda')

    RESULTS_DIR = args.results_path
    ADD_NOISE = args.add_noise
    n_hidden = args.n_hidden
    window = 100
    ch = 93

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

    train_total_data, train_size, _, _, test_data1, test_labels1 = mnist_data.get_hmd_data(test_id=args.test_id, dataset_dir='HDM_data/')

    n_samples = train_size
    print("n_samples ",n_samples)


    """ create network """
    keep_prob = 0.99
    encoder = vae.Encoder(dim_img, n_hidden, dim_z, keep_prob).to(device)
    decoder = vae.Decoder(dim_z, n_hidden, dim_img, keep_prob).to(device)

    # + operator will return but .extend is inplace no return.
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learn_rate)

    enc_scheduler = StepLR(optimizer, step_size=200, gamma=0.5)

    """ training """
    # Plot for reproduce performance
    if PRR:
        PRR = plot_utils.Plot_Reproduce_Performance(RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, IMAGE_SIZE_WMNIST,
                                                    IMAGE_SIZE_HMNIST, PRR_resize_factor)

        test_datat = np.load("hmd_loss1_200_ae_action_fold1_rnd1_14.npy")  #load test data

        test_datast = np.load("hmd_loss1_200_ae_action_fold1_rnd1_test.npy") #load original test data

        test_data = test_datast[:test_datat.shape[0],:] 

        test_datat = test_datat.reshape(test_datat.shape[0], -1)

        x_PRR = test_datat[:, :].copy()

        print('performed')  


        i,j = 0,100
        plt.figure(figsize = (10,5))
        x_graph = x_PRR
        plt.plot(x_graph[0,:j],'b',label='Original')
        plt.plot(x_graph[4,:j],'b',label='Original')
        plt.plot(x_graph[10,:j],'b',label='Original')
        plt.plot(x_graph[11,:j],'b',label='Original')

        x_PRR_o = test_data.copy()


        ##############
        if ADD_NOISE > 0:  



            print('add noise', ADD_NOISE)  

            x_graph = x_PRR
            plt.plot(x_graph[0,:j],'r',label='Noise Added')
            plt.plot(x_graph[4,:j],'r',label='Noise Added')
            plt.plot(x_graph[10,:j],'r',label='Noise Added')
            plt.plot(x_graph[11,:j],'r',label='Noise Added')

            print("x_PRR.shape ",x_PRR.shape) #100,1500

        plt.title('Input')
        #plt.savefig('Input.jpg')

    best_acc = 0.0
    
    # load train data
    total_batch = int(n_samples / batch_size)
    min_tot_loss = np.inf
    for epoch in range(n_epochs):

        train_datat = np.load("recoveredTrain_19.npy") #load recovered train data from 2nd step

        train_datat = train_datat.reshape(train_datat.shape[0], -1)

        s = np.arange(train_datat.shape[0])
        np.random.shuffle(s)

        train_data_ = train_datat[s,:].copy()

        train_datatr = np.load("hmd_loss1_200_ae_action_fold1_train.npy") #load original train data

        train_datatr = train_datatr.reshape(train_datat.shape[0], -1)
        train_data_target = train_datatr[s,:].copy()

        train_datalabel = np.load("hmd_loss1_200_ae_action_fold1_train_labels.npy") #load train data label

        train_data_label = train_datalabel[s,:].copy()


        # Loop over all batches
        encoder.train()
        decoder.train()
        for i in range(total_batch):
            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (n_samples)
            batch_xs_input = train_data_[offset:(offset + batch_size), :]

            batch_xs_target = train_data_target[offset:(offset + batch_size), :]

            batch_xs_inputa = batch_xs_input.reshape((batch_xs_input.shape[0],window,ch))
            batch_xs_input = np.transpose(batch_xs_inputa, (0, 2, 1)) 

            batch_xs_targeta = batch_xs_target.reshape((batch_xs_target.shape[0],window,ch))
            batch_xs_target = np.transpose(batch_xs_targeta, (0, 2, 1)) 
            batch_xs_input, batch_xs_target = torch.from_numpy(batch_xs_input).float().to(device), \
                                              torch.from_numpy(batch_xs_target).float().to(device)

            assert not torch.isnan(batch_xs_input).any()
            assert not torch.isnan(batch_xs_target).any()

            y, z, tot_loss, loss_likelihood, loss_divergence = \
                                        vae.get_loss_ae(encoder, decoder, batch_xs_input, batch_xs_target) #AE loss function

            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

        print("epoch %d: L_tot %04.6f L_likelihood %04.6f L_divergence %04.6f" % (
                                                epoch, tot_loss.item(), loss_likelihood.item(), loss_divergence.item()))




        encoder.eval()
        decoder.eval()


        y_PRR_t = np.zeros((x_PRR.shape[0], ch, window))

        y_PRR_tr = np.zeros((train_data_.shape[0], ch, window))

        # if minimum loss is updated or final epoch, plot results
        if min_tot_loss > tot_loss.item() or epoch + 1 == n_epochs:
            min_tot_loss = tot_loss.item()

            if args.save == True and (epoch > 400 or epoch % 300 == 0) and epoch > 0:
                save(args.results_path + "/",encoder, decoder, optimizer, epoch)

            # Plot for reproduce performance
            if PRR:
                
                for i in range(total_batch):
                    # Compute the offset of the current minibatch in the data.
                    offset = (i * batch_size) % (x_PRR.shape[0])
                    batch_xs_input = x_PRR[offset:(offset + batch_size), :] #######original

                    batch_xs_target = test_data[offset:(offset + batch_size), :]

                    batch_xs_inputa = batch_xs_input.reshape((batch_xs_input.shape[0],window,ch))
                    batch_xs_input = np.transpose(batch_xs_inputa, (0, 2, 1)) 

                    batch_xs_targeta = batch_xs_target.reshape((batch_xs_target.shape[0],window,ch))
                    batch_xs_target = np.transpose(batch_xs_targeta, (0, 2, 1)) 


                    batch_xs_input, batch_xs_target = torch.from_numpy(batch_xs_input).float().to(device), \
                                                     torch.from_numpy(batch_xs_target).float().to(device)

                    assert not torch.isnan(batch_xs_input).any()
                    assert not torch.isnan(batch_xs_target).any()

                    x_PRR_t = batch_xs_input.reshape(batch_xs_input.shape[0], ch, -1)
                    get_output = vae.get_ae(encoder, decoder, x_PRR_t)

                    y_PRR_t[offset:(offset + batch_size), :] = get_output.detach().cpu().numpy() 

               

                x_PRR_t = x_PRR.reshape(x_PRR.shape[0], ch, -1)
                y_PRR = y_PRR_t.reshape(y_PRR_t.shape[0], -1)

                if epoch > 20 or epoch % 30 == 0:

                    Xt = x_PRR_o.reshape(x_PRR_o.shape[0],window, ch) #num, ch, hei, wid
                    Xt = np.transpose(Xt, (0, 2, 1)) 
                    
                    Yt = y_PRR_t.reshape(y_PRR.shape[0],ch, window)
                    Yt = Yt.reshape(Yt.shape[0],ch, window)
                    Yt = np.float32(Yt)

                    PSNR_v = calc_psnr(Variable(torch.tensor(Xt).float()),Variable(torch.tensor(Yt).float()))

                    #PSNR_v64 = calc_psnr(Variable(torch.tensor(Xt[:64]).float()),Variable(torch.tensor(Yt[:64]).float()))
                    #PSNR_v128 = calc_psnr(Variable(torch.tensor(Xt[:128]).float()),Variable(torch.tensor(Yt[:128]).float()))
                    #PSNR_v256 = calc_psnr(Variable(torch.tensor(Xt[:256]).float()),Variable(torch.tensor(Yt[:256]).float()))

                    if best_acc < PSNR_v.item():
                        best_acc = PSNR_v.item()

                    print("-------------test--------------") 
                    print("PSNR_v: ",PSNR_v.item(),"PSNR_best: ",best_acc) 
                    #print("PSNR_v64: ",PSNR_v64.item()) 
                    #print("PSNR_v128: ",PSNR_v128.item()) 
                    #print("PSNR_v256: ",PSNR_v256.item()) 
                    print("-------------------------------") 



                name= args.results_path + "/PRR_x_epoch_%02d" % (epoch) + ".jpg"


                i,j = 0,100

                plt.figure(figsize = (10,5))

                x_graph = x_PRR_o 
                y_graph = y_PRR

                plt.plot(x_graph[0,:j],'b',label='Original')
                plt.plot(x_graph[4,:j],'b',label='Original')

                plt.plot(x_graph[10,:j],'b',label='Original')
                plt.plot(x_graph[11,:j],'b',label='Original')


                plt.plot(y_graph[0,:j],'r',label='Generated')
                plt.plot(y_graph[4,:j],'r',label='Generated')

                plt.plot(y_graph[10,:j],'r',label='Generated')
                plt.plot(y_graph[11,:j],'r',label='Generated')



                name1="result_x_%02d" % (epoch)
                plt.title(name1)

                plt.figure(figsize = (10,5))


                name= args.results_path + "/PRR_y_epoch_%02d" % (epoch) + ".jpg"


                plt.plot(x_graph[0,500:500+j],'y',label='Original')
                plt.plot(x_graph[4,500:500+j],'y',label='Original')
                plt.plot(x_graph[10,500:500+j],'y',label='Original')
                plt.plot(x_graph[11,500:500+j],'y',label='Original')


                plt.plot(y_graph[0,500:500+j],'g',label='Generated')
                plt.plot(y_graph[4,500:500+j],'g',label='Generated')
                plt.plot(y_graph[10,500:500+j],'g',label='Generated')
                plt.plot(y_graph[11,500:500+j],'g',label='Generated')

                name1="result_y_%02d" % (epoch)
                plt.title(name1)


            if epoch > 30:

                for i in range(total_batch):
                    # Compute the offset of the current minibatch in the data.
                    offset = (i * batch_size) % (x_PRR.shape[0])
                    batch_xs_input = train_data_[offset:(offset + batch_size), :] #######original

                    batch_xs_inputa = batch_xs_input.reshape((batch_xs_input.shape[0],window,ch))
                    batch_xs_input = np.transpose(batch_xs_inputa, (0, 2, 1)) 

                    batch_xs_input = torch.from_numpy(batch_xs_input).float().to(device)

                    assert not torch.isnan(batch_xs_input).any()

                    x_PRR_t = batch_xs_input.reshape(batch_xs_input.shape[0], ch, -1)
                    get_output = vae.get_ae(encoder, decoder, x_PRR_t)

                    y_PRR_tr[offset:(offset + batch_size), :] = get_output.detach().cpu().numpy() 

               
                y_PRR_trsave = y_PRR_tr.reshape(y_PRR_tr.shape[0], -1)


                if (epoch > 20 or epoch % 30 == 0) and epoch >= 0:

                    Yt = y_PRR_t.reshape(y_PRR_t.shape[0],ch, window)
                    Yt = np.transpose(Yt, (0, 2, 1)) 
                    Yt = np.float32(Yt)
                    y_PRR_save = Yt.reshape(Yt.shape[0],window,ch)
                    print("y_PRR_save",y_PRR_save.shape)

                    mdic2 = {"reconstructed_actions_optimized_z": y_PRR_save}

                    savemat(str(args.results_path)+'/refine3rd_'+str(epoch)+'_output.mat', mdic2)
                    np.save(str(args.results_path)+'/refine3rd_'+str(epoch)+'_output',y_PRR_save)

                    Yt2 = y_PRR_trsave.reshape(y_PRR_trsave.shape[0],ch, window)
                    Yt2 = np.transpose(Yt2, (0, 2, 1)) 




if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)
