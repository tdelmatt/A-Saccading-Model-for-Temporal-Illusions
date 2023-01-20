# Model definition
from typing import List, Tuple, Optional
from yacs.config import CfgNode as CN
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F_vis
from vgg import vgg11_bn
import pytorch_ssim


class SSIMMSE(nn.Module):

    def __init__(self, device):
        # device ain't helpful at this point
        super().__init__()
        self.ssim = pytorch_ssim.SSIM(window_size=7)
        # self.ssim = pytorch_ssim.SSIM(window_size=11)

    def forward(self, x, y):
        # x, y: T B C H W
        x = x.flatten(0, 1)
        y = y.flatten(0, 1)
        return self.ssim(x, y)


class CIFARMSE(nn.Module):

    # returns mse of activations on nth layer of a CIFAR module
    def __init__(self, device, layer=4):
        # device ain't helpful at this point
        super().__init__()
        # self.net = mobilenet_v2(pretrained=True).features[:layer]
        self.net = vgg11_bn(pretrained=True).features[:layer]
        self.layers = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        self.net.eval()
        self.mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(-1, 1, 1)
        self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(-1, 1, 1)
        # self.mse = nn.MSELoss()

    def forward(self, x, y):
        # x, y: T B C H W
        # mobile net needs 3D inputs in 0-1. We have 1D -1 to 1
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        x = x.flatten(0, 1).expand(-1, 3, -1, -1)
        y = y.flatten(0, 1).expand(-1, 3, -1, -1)
        x = x / 2. + 0.5
        y = y / 2. + 0.5
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        losses = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            y = layer(y)
            if self.layers[i] != "M":
                losses.append(torch.nn.functional.l1_loss(x, y))
        return torch.stack(losses).mean()


class GaussianFourierFeatureTransform(nn.Module):

    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    * Tune scale higher to repr higher frequency features
    """
    def __init__(self, num_input_channels=2, mapping_size=64, scale=1):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self.register_buffer("_B", torch.randn((num_input_channels, mapping_size)) * scale)

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class PolarTransform(nn.Module):

    def __init__(self):
        super().__init__()

    def _cart_to_pol(self, x, y):
        rho = torch.sqrt(x**2 + y**2)
        phi = torch.atan2(y, x) / np.pi
        return(rho, phi)

    def forward(self, grid):
        batch, channel, x, y = grid.shape
        grid = grid - 0.5
        item = self._cart_to_pol(grid[0,0], grid[0, 1])
        return torch.stack(item, 0).unsqueeze(0)


class SaccadingRNN(pl.LightningModule):
    # We'll train this network with images. We can saccade for e.g. 100 timesteps per image and learn through self-supervision.
    # This model is driven by a saccading policy (here it is hardcoded to pick a random sequence of locations)
    # The model is trained to predict the next visual input.
    # 3 loss schemes
    # 1. E2E, deconv RNN state to predict pixels

    def __init__(
        self,
        config: CN,
    ):
        self.cfg = config.MODEL
        assert self.cfg.TYPE is 'gru', 'non-gru rnns unsupported'
        super().__init__()
        conv_dim = self.cfg.CONV_CHANNELS
        # self.conv_outw = self.conv_outh = 4 # Manually checked
        self.conv_outw = self.conv_outh = 2 # Manually checked

        n_pixels = self.cfg.FOV_HEIGHT * self.cfg.FOV_WIDTH
        n_channels_image = config.TASK.CHANNELS
        RNN_hidden_size = self.cfg.ADAPTATION_HIDDEN_SIZE

        self.sparsity_tracking_dir = 'sparsity_tracking/'

        if self.cfg.ADAPTATION_LAYER:
            print("adaptation type is {}".format(self.cfg.ADAPTATION_TYPE))
            if self.cfg.ADAPTATION_SPARSITY_LOSS:

                #make directories to save sparsity tracking images
                outer_d_name = 'sparsity_tracking/'
                if not os.path.isdir(outer_d_name):
                    os.mkdir(outer_d_name)
                
                outer_d_name = 'sparsity_tracking/' + self.cfg.ADAPTATION_TYPE + '/'
                if not os.path.isdir(outer_d_name):
                    os.mkdir(outer_d_name)
                
                for i in range(10000):
                    dname = outer_d_name +'run_' + str(i) + '/'
                    if not os.path.isdir(dname):
                        os.mkdir(dname)
                        self.sparsity_tracking_dir = dname
                        break

                self.sparsity_tracking_list = []

                #create sparsity loss weighting matrix...idea here is that pixels
                #towards the center have no loss
                height_pixels = self.cfg.FOV_HEIGHT
                width_pixels = self.cfg.FOV_WIDTH
                h_center = height_pixels / 2
                w_center = width_pixels / 2

                self.sparsity_loss_weighting_matrix =  torch.zeros((height_pixels,width_pixels), device = self.device)
                self.distance_matrix = torch.zeros((height_pixels,width_pixels), device = self.device)
                print("distance mat device {}".format(self.device))
                self.do_not_change_pixel_list = []


                for h in range(height_pixels):
                    for w in range(width_pixels):
                        
                        h_dist = abs(h + 1 - 16)
                        w_dist = abs(w + 1 - 16)
                        
                        #distance from center
                        total_dist = (h_dist ** 2 +  w_dist **2) ** .5
                        self.distance_matrix[h,w] = total_dist

                        #pixels within 
                        if total_dist < self.cfg.ADAPTATION_SPARSITY_NO_LOSS_DISTANCE:
                            self.sparsity_loss_weighting_matrix[h,w] = 0
                            self.do_not_change_pixel_list.append((h,w))
                        else:
                            self.sparsity_loss_weighting_matrix[h,w] = \
                            (total_dist ** self.cfg.ADAPTATION_SPARSITY_EXPONENT) *\
                             self.cfg.ADAPTATION_SPARSITY_LOSS_WEIGHT
                

                spnp = self.sparsity_loss_weighting_matrix.cpu().numpy()
                savedir = self.sparsity_tracking_dir + 'sparsity_weight_.png'

                #plt.imsave(savedir, spnp)
                plt.matshow(spnp)
                plt.colorbar()
                plt.savefig(savedir)
                plt.clf()


            if self.cfg.ADAPTATION_TYPE == 'multi_RNN':

                self.adapt_linear = nn.Linear(RNN_hidden_size * n_channels_image, n_channels_image)
                self.adapt_activation = nn.LeakyReLU()

                self.adaptation_RNN_list = nn.ModuleList()#nn.ParameterList()
                for pixel in range(n_pixels):
                    self.adaptation_RNN_list.append(nn.RNN(input_size =  n_channels_image, 
                        hidden_size = RNN_hidden_size))

            elif self.cfg.ADAPTATION_TYPE == 'single_RNN':
                self.adaptation_RNN = nn.RNN(input_size =  n_channels_image + 1, 
                      hidden_size = RNN_hidden_size)
                
                self.adapt_linear = nn.Linear(RNN_hidden_size, n_channels_image)
                self.adapt_activation = nn.ReLU()

            elif self.cfg.ADAPTATION_TYPE == 'feature_activation':
                self.adapt_linear = nn.Linear(n_channels_image + 1, 1)
                self.adapt_activation = nn.Sigmoid()

            elif self.cfg.ADAPTATION_TYPE == 'hardcoded_v1':
                pass
        
        
        conv_dim = self.cfg.CONV_CHANNELS
        self.conv_outw = self.conv_outh = 2 # Manually checked
        self.loss_tracking = []
        self.sparsity_loss_tracking = []

        def gen_activation():
            return nn.ReLU() if self.cfg.ACTIVATION is 'relu' else nn.LeakyReLU(0.05)

        self.cnn_sensory = nn.Sequential(
            conv(config.TASK.CHANNELS, conv_dim, 4),
            gen_activation(),
            conv(conv_dim, conv_dim, 4),
            gen_activation(),
            conv(conv_dim, conv_dim * 2, 4),
            gen_activation(),
            conv(conv_dim * 2, conv_dim, 4),
            gen_activation(),
        )
        flat_cnn_out = conv_dim * self.conv_outh * self.conv_outw

        self.proprio_size = 2
        self.proprio_grid = None
        if self.cfg.POLAR_PROPRIO:
            self.proprio_transform = PolarTransform()
        if self.cfg.FOURIER_PROPRIO:
            self.proprio_size = 32
            self.proprio_transform = GaussianFourierFeatureTransform(mapping_size=self.proprio_size // 2)

        if self.cfg.INCLUDE_PROPRIO:
            memory_input_size = self.cfg.SENSORY_SIZE - self.proprio_size
        else:
            memory_input_size = self.cfg.SENSORY_SIZE
        if self.cfg.TRANSITION_MLP:
            self.sensory_to_memory = nn.Sequential(
                nn.Linear(flat_cnn_out, memory_input_size),
                gen_activation(),
                nn.Linear(memory_input_size, memory_input_size)
            )
        else:
            self.sensory_to_memory = nn.Linear(flat_cnn_out, memory_input_size)

        if self.cfg.INCLUDE_PROPRIO:
            sensory_prediction_size = self.cfg.HIDDEN_SIZE + self.proprio_size
        else:
            sensory_prediction_size = self.cfg.HIDDEN_SIZE

        # Extract cnn_view prediction from memory
        self.predict_sensory = nn.Sequential(
            nn.Linear(sensory_prediction_size, self.cfg.SENSORY_SIZE),
            gen_activation(),
            nn.Linear(self.cfg.SENSORY_SIZE, flat_cnn_out),
            nn.Unflatten(-1, (conv_dim, self.conv_outh, self.conv_outw))
        )
        self.rnn = nn.GRU(self.cfg.SENSORY_SIZE, self.cfg.HIDDEN_SIZE, 1)

        predictive_output = config.TASK.CHANNELS if self.cfg.QUANTIZED_RECONSTRUCTION <= 1 else self.cfg.QUANTIZED_RECONSTRUCTION
        if self.cfg.UPSAMPLE_CONV:
            self.cnn_predictive = nn.Sequential(
                upsample_conv(conv_dim, conv_dim * 2, 2),
                gen_activation(),
                upsample_conv(conv_dim * 2, conv_dim, 2),
                gen_activation(),
                upsample_conv(conv_dim, conv_dim, 2),
                gen_activation(),
                upsample_conv(conv_dim, predictive_output, 2),
                nn.Tanh()
            )
        else:
            self.cnn_predictive = nn.Sequential(
                deconv(conv_dim, conv_dim * 2, 4),
                gen_activation(),
                deconv(conv_dim * 2, conv_dim, 4),
                gen_activation(),
                deconv(conv_dim, conv_dim, 4),
                gen_activation(),
                deconv(conv_dim, predictive_output, 4),
                nn.Tanh()
            )
        self.mse = nn.MSELoss()


        if self.cfg.CIFAR_LOSS:
            self.perceptual_mse = CIFARMSE(self.device)
        elif self.cfg.SSIM_LOSS:
            self.perceptual_mse = SSIMMSE(self.device)
        else:
            self.perceptual_mse = self.mse
        self.xent = nn.CrossEntropyLoss()


        if 'contrast' in self.cfg.OBJECTIVES:
            self.contrast = nn.Sequential(
                nn.Linear(2 * memory_input_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )


        if 'adversarial_patch' in self.cfg.OBJECTIVES:
            self.patch_contrast = nn.Sequential(
                nn.Linear(flat_cnn_out, 1)
            )
            self.automatic_optimization = False

        self.weight_decay = config.TRAIN.WEIGHT_DECAY
        self.saccade_training_mode = self.cfg.SACCADE
        self.view_mask = self._generate_falloff_mask()

    def _gen_proprio(self, target):
        # proprio_grid is pre-calculated coordinate transform
        if not (self.cfg.FOURIER_PROPRIO or self.cfg.POLAR_PROPRIO):
            self.proprio_grid = 0 # dummy
            return
        coords = np.linspace(0, 1, target.shape[2], endpoint=False)
        xy_grid = np.stack(np.meshgrid(coords, coords), -1)
        proprio_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(self.device)
        self.proprio_grid = self.proprio_transform(proprio_grid).squeeze(0) # C H W

    def _transform_proprio(self, proprio):
        # b x 2 -> b x h
        if not (self.cfg.FOURIER_PROPRIO or self.cfg.POLAR_PROPRIO):
            return proprio
        flat_prop = proprio.flatten(0, 1)
        #print(transformed)
        #print(self.proprio_grid.size())
        #print(flat_prop.size())
        transformed = self.proprio_grid[:, flat_prop[:, 0], flat_prop[:, 1]].permute(1, 0)

        return transformed.unflatten(0, proprio.size()[:2])

    def _evaluate_image_pred(self, gt, pred):
        # Note: the actual quantization scheme was proposed in a more sophisticated way (ref: https://github.com/richzhang/colorization/tree/caffe)
        # Doing this for simplicity
        # gt: T B C H W
        if self.cfg.QUANTIZED_RECONSTRUCTION <= 1:
            return self.perceptual_mse(pred, gt)
        # Expecting GT to vary [-0.5, 0.5]
        quantized_gt = ((gt + 0.5) * (self.cfg.QUANTIZED_RECONSTRUCTION - 1)).long().flatten(0, 2)
        return self.xent(pred.flatten(0, 1), quantized_gt)

    def _quantized_pred(self, pred):
        if self.cfg.QUANTIZED_RECONSTRUCTION <= 1:
            return pred
        # Expecting GT to vary [-0.5, 0.5]
        with torch.no_grad():
            quantized_pred = torch.argmax(pred, dim=2).float() # T B H W varying 0 - 7
            quantized_pred = quantized_pred / (self.cfg.QUANTIZED_RECONSTRUCTION - 1) # asymmetry is nervewracking
            BIN_SPAN = 0.5 / self.cfg.QUANTIZED_RECONSTRUCTION
            quantized_pred = quantized_pred - 0.5 + BIN_SPAN # T B H W
            return quantized_pred.unsqueeze(-3) # T B C H W

    def _generate_saccades(self, image, length=50, mode=None, fixate_drift = .03) -> torch.tensor:
        # Generate a sequence of saccading focal coordinates.
        # * Does not provide margin, i.e. coordinates can be on corner of image.
        # * We allow this so model can observe corner accurately.
        # args:
        #   image: [* x H x W]. image or image batch to saccade over.
        #   length: length of saccading sequence
        #   mode: saccadding mode. (controlling model operation)
        # returns:
        #   coords: [length x B x 2] . pixel coordinates
        if mode is None:
            mode = self.saccade_training_mode
        H, W = image.size()[-2:]

        if mode == 'random': # uniform distribution
            SACCADE_BATCH = self.cfg.SACCADE_BATCH
            coords_ratio = torch.rand((length, SACCADE_BATCH, 2), device=image.device)
        elif mode == 'walk': # random walk
            # TODO Make walking more realistic -- humans don't random walk off the image.
            # A better heuristic is to weigh directional probability by location
            #   but that might be too slow.
            WALK_PACE = self.cfg.WALK_PACE
            SACCADE_BATCH = self.cfg.SACCADE_BATCH
            start_ratio = torch.tensor([0.5, 0.5], device=self.device).expand(1, SACCADE_BATCH, -1) # 1 x B x 2
            deltas_ratio = torch.randn((length - 1, SACCADE_BATCH, 2), device=self.device) * WALK_PACE
            coords_ratio = torch.cat([start_ratio, deltas_ratio], dim=0)
            coords_ratio = torch.cumsum(coords_ratio, dim=0)
        elif mode == 'fixate': # center with Gaussian offset for drift.
            # ! Untested
            start = torch.tensor([0.5, 0.5], device=self.device).unsqueeze(0).float() # 1 x 2
            DRIFT = fixate_drift # 0.05 of a 128 x 128 would constrain to a 6x6 range, seems reasonable.
            # TODO Chris - add config to control amount of drift
            # TODO Chris - calculate an appropriately calibrated fixation radius based on human measurements (at level of saccadic drift)
            deltas = torch.randn((length, 2), device=self.device) * DRIFT
            coords_ratio = torch.tensor(start + deltas)
        elif mode == 'constant':
            coords_ratio = torch.full((length, 2), 0.5, device=self.device)
        coords_ratio = torch.clamp(coords_ratio, 0, 1)
        if self.cfg.FOURIER_PROPRIO or self.cfg.POLAR_PROPRIO:
            # fix off by one error that only raises in this case
            image_scale = torch.tensor([H - 1, W - 1], device=self.device).float()
        else:
            image_scale = torch.tensor([H, W], device=self.device).float()
        coord = (coords_ratio * image_scale).long()
        if len(coord.size()) == 2:
            return coord.unsqueeze(1)
        return coord

    def _generate_falloff_mask(self):
        # TODO Joel look at what Gaussian blur actually is.
        # Construct FOV falloff (H x W)
        # Since we only construct this once, inefficient manual construction is fine
        mask = torch.zeros(
            self.cfg.FOV_WIDTH, self.cfg.FOV_HEIGHT, device=self.device
        ).float()
        center_h = (mask.size(-2) - 1) / 2.
        center_w = (mask.size(-1) - 1) / 2.
        for h in range(mask.size(-2)):
            for w in range(mask.size(-1)):
                mask[h,w] = (h - center_h) ** 2 + (w - center_w) ** 2
        return mask

    def _predict_at_location(self, state, focus, mode='patch'):
        r"""
        Memory conditioned predictions
        Args:
            state: [T x B x H] state to make prediction with (either CNN output in predictive coding, or memory)
            focus: [T x B x 2] location to predict at (given as pixel locs (long))
            mode: 'patch' for pixel reconstruction or 'percept' for CNN reconstruction
        Returns:
            patches: [B x C x H x W] predicted patches at given locations
        """
        if self.cfg.INCLUDE_PROPRIO:
            focus = self._transform_proprio(focus) # T x 2 -> T x H
            if len(focus.size()) == 2:
                focus = focus.unsqueeze(1).expand(-1, state.size(1), -1)
            state = torch.cat([state, focus], dim=-1)
        prediction_cue = self.predict_sensory(state) # extract cnn prediction
        if mode == 'patch': # Make a prediction about image patch from memory
            # T x B x Hidden -> T x B x C x H x W
            return self.cnn_predictive(prediction_cue.flatten(0, 1)).unflatten(
                0, (state.size(0), state.size(1))
            )
        else:
            return prediction_cue

    def sparsity_tracking(self, lower_thresh, upper_thresh, activations_matrix, imsave_frequency = 1,
        sparsity_loss = None, sparsity_mat = None):
        
        activations = torch.zeros(activations_matrix.size(), device = self.device)
        non_activations = torch.zeros(activations_matrix.size(), device = self.device)
        

        activations[(activations_matrix >= upper_thresh)] = 1
        non_activations[(activations_matrix <= lower_thresh)] = 1

        percent_active = torch.sum(activations) / torch.numel(activations_matrix)
        percent_inactive = torch.sum(non_activations) / torch.numel(activations_matrix)

        #append to sparsity tracking list
        self.sparsity_tracking_list.append([percent_active, percent_inactive])

        val = np.random.random()
        if val < imsave_frequency:

            active_np = activations.cpu().numpy()
            inactive_np = non_activations.cpu().numpy()

            mean_active = np.mean(active_np, axis = (0,1,2))
            mean_inactive = np.mean(inactive_np, axis = (0,1,2))
            mean_raw_activations = np.mean(activations_matrix.cpu().detach().numpy(), axis = (0,1,2))

            save_dir1 = self.sparsity_tracking_dir + 'activations_'
            save_dir2 = self.sparsity_tracking_dir + 'inactive_'
            save_dir3 = self.sparsity_tracking_dir + 'raw_activations_'
            save_dir4 = self.sparsity_tracking_dir + "sparsity_mat_resized"
            save_dir5 = self.sparsity_tracking_dir + "sparsity_loss"

            """
            self.save_image_as_matrix(self.sparsity_tracking_dir + "sparsity_mat_resized.png"
                        reshaped_sparsity_matrix[0,0,0,:,:].cpu().numpy())
                
            self.save_image_as_matrix(self.sparsity_tracking_dir + "sparsity_loss.png",
                        torch.mean(sp_loss_mat, dim = (0,1,2)).detach().cpu().numpy())
            """

            for i in range(100):
                impath = save_dir1 + str(i) + '.png'
                if not os.path.isfile(impath):
                    impath2 = save_dir2 + str(i) + '.png'
                    impath3 = save_dir3 + str(i) + '.png'
                    impath4 = save_dir4 + str(i) + '.png'
                    impath5 = save_dir5 + str(i) + '.png'
                    plt.imsave(impath, mean_active)
                    plt.imsave(impath2, mean_inactive)
                    plt.imsave(impath3, mean_raw_activations)

                    if sparsity_mat is not None:
                        self.save_image_as_matrix(impath4,
                            sparsity_mat[0,0,0,:,:].cpu().numpy())

                    if sparsity_loss is not None:
                        self.save_image_as_matrix(impath5,
                            torch.mean(sparsity_loss, dim = (0,1,2)).detach().cpu().numpy())

                    self.save_image_as_matrix(impath3, mean_raw_activations)
                    break

    def save_image_as_matrix(self, file_path, matrix):
        plt.matshow(matrix)
        plt.colorbar()
        plt.savefig(file_path)
        plt.clf()

    def forward(self, view, saccade_focus, hidden_state):
        r"""
            Roll forward perception.
            For efficiency, this is a batch update.
            Requires some reshaping for a single step (and thus may be annoying for active perception)
            Args:
                view: [T x B x C x H x W] - cropped views.
                saccade_focus: [T x B' x 2] - proprioception (shared across batch)
                hidden_state: [B x Hidden] (initial hidden state)
            Returns:
                cnn_view: [T x B x C x H' x W'] CNN outputs
                memory_prediction: [T x B x [C x H' x W']] RNN outputs
                hidden_state: [B x Hidden] Final hidden state
        """

        if self.cfg.ADAPTATION_LAYER:

            olshausen_exp_loss = lambda x: -torch.exp(-7*(1*(x + .5)**2))
            olshausen_v2_loss = lambda x: -torch.exp(-.1*(1*(x + 6)**2)) + .003*x
            modified_l1 = lambda x: (-20*torch.abs((x -.5))+ 10)
            olshausen_v3_loss = lambda x: (-10*torch.exp(-.02*(1*(x + 12)**2)) + .1*(x + 6) + 10)
            #proposed augmented olshausen loss
            #not tested
            #olshausen_exp_loss = lambda x: -torch.exp(-7*(1*(x + .5)**2)) + .02*x

            #print("adaptation type is {}".format(self.cfg.ADAPTATION_TYPE))
            if self.cfg.ADAPTATION_TYPE == 'multi_RNN':

                rnn_out_list = []

                #for all rnns, slice view to obtain pixel, 
                for i, rnn in enumerate(self.adaptation_RNN_list):
                    out_rnn, temp_hidden = rnn(view.flatten(3)[:,:,:,i])
                    #print("out rnn size {}".format(out_rnn.size()))

                    if self.cfg.ADAPTATION_HIDDEN_SIZE > 1:
                        out_linear = self.adapt_linear(out_rnn)

                        #sparsity loss on linear outputs
                        sparsity_loss = torch.mean(torch.abs(out_linear + 1))

                        #take sparsity over these activations
                        out_adapt = self.adapt_activation(out_linear)

                    rnn_out_list.append(out_adapt.reshape(view.size()[0], 
                        view.size()[1], view.size()[2], 1))

                #after cat T x B x C x HW
                adapt_out_view = torch.cat(rnn_out_list, dim = 3)

                #after unflatten T x B x C x H x W
                adapt_out_view = adapt_out_view.unflatten(3, (view.size()[3], view.size()[4]))

                #assert output size is not modified
                print("a out view size {}".format(adapt_out_view.size()))
                print("normal view size {}".format(view.size()))
                assert(adapt_out_view.size() == view.size())

            elif self.cfg.ADAPTATION_TYPE == 'single_RNN':
                #print("single running!!!")

                #input size T x B x C x H x W
                #permute to T x B x H x W x C
                view1 = view.permute(0,1,3,4,2)

                #we want T x BHW x C
                rnn_input = view1.flatten(1,3)

                #reshape 1 x 1 x H x W x 1
                reshaped_distance_matrix = self.distance_matrix.reshape(1,1,
                        self.distance_matrix.size()[0], self.distance_matrix.size()[1] ,1).to(self.device)
                
                #repeat to T x B x H x W x 1
                repeated_distance_matrix = reshaped_distance_matrix.repeat(
                    view1.size()[0], view1.size()[1],1,1,1)

                #we want T x BHW x 1
                repeated_distance_matrix = repeated_distance_matrix.flatten(1,3)

                conc_rnn_input = torch.cat([rnn_input, repeated_distance_matrix], dim = 2)

                #run through rnn
                rnn_output, hidden_output = self.adaptation_RNN(conc_rnn_input)

                #linear
                linear_output = self.adapt_linear(rnn_output)
                
                #unflatten T x BHW x C -> T x B x H x W x C
                linear_output = linear_output.unflatten(1, (view1.size()[1], view1.size()[2], view1.size()[3]))
                
                #permute T x B x H x W x C to T x B x C x H x W
                linear_output = linear_output.permute(0,1,4,2,3)

                #reshape sparsity matrix 1,1,1,H,W
                reshaped_sparsity_matrix = self.sparsity_loss_weighting_matrix.reshape(1,
                        1,1, self.sparsity_loss_weighting_matrix.size()[0], self.sparsity_loss_weighting_matrix.size()[1]).to(self.device)

                #sparsity loss on linear outputs
                if self.cfg.ADAPTATION_SPARSITY_LOSS_TYPE == 'L1':
                    sparsity_loss = torch.mean(torch.abs((linear_output + 1) * reshaped_sparsity_matrix))

                elif self.cfg.ADAPTATION_SPARSITY_LOSS_TYPE == 'olshausen':
                    sparsity_loss = torch.mean(olshausen_exp_loss(linear_output) * reshaped_sparsity_matrix)

                elif self.cfg.ADAPTATION_SPARSITY_LOSS_TYPE == 'olshausen_v2_loss':
                    sparsity_loss = torch.mean(olshausen_v2(linear_output) * reshaped_sparsity_matrix)

                elif self.cfg.ADAPTATION_SPARSITY_LOSS_TYPE == 'olshausen_v3_loss':
                    sp_loss_mat = olshausen_v3_loss(linear_output) * reshaped_sparsity_matrix
                    sparsity_loss = torch.mean(sp_loss_mat)

                else:
                    raise Exception("sparsity loss not recognized")

                #relu or leaky relu
                adapt_out_view = self.adapt_activation(linear_output)

                val = np.random.random()

                #sparsity tracking call
                if val < self.cfg.SPARSITY_TRACKING_FREQUENCY:

                    self.sparsity_tracking(0, .1, adapt_out_view, self.cfg.SPARSITY_IMAGE_TRACKING_FREQUENCY,
                                    sparsity_loss = sp_loss_mat, sparsity_mat = reshaped_sparsity_matrix)

                assert(adapt_out_view.size() == view.size())


            elif self.cfg.ADAPTATION_TYPE == 'feature_activation':

                print("RUNNING FEATURE ACTIVATION ADAPTATION LAYER")
                #input size T x B x C x H x W
                T = view.size()[0]

                previous_4_mean = torch.zeros(view.size(), device = self.device)

                previous_4_mean[0,:,:,:,:] = .5
                previous_4_mean[1,:,:,:,:] = torch.mean(view[0:1,:,:,:,:], dim = 0)
                previous_4_mean[2,:,:,:,:] = torch.mean(view[0:2,:,:,:,:], dim = 0)
                previous_4_mean[3,:,:,:,:] = torch.mean(view[0:3,:,:,:,:], dim = 0)

                #want to take the mean of previous 4 timesteps for each pixel
                for t in range(4, T):
                    previous_4_mean[t,:,:,:,:] = torch.mean(view[t-4: t,:,:,:,:], dim = 0)
                #print("")

                #T x B x C x H x W
                input_feature_1 = torch.abs(previous_4_mean - view)

                #input size T x B x C x H x W
                #permute to T x B x H x W x C
                input_feature_1 = input_feature_1.permute(0,1,3,4,2)

                #we want T x BHW x C
                input_feature_1 = input_feature_1.flatten(1,3)

                #reshape 1 x 1 x H x W x 1
                reshaped_distance_matrix = self.distance_matrix.reshape(1,1,
                        self.distance_matrix.size()[0], self.distance_matrix.size()[1] ,1).to(self.device)
                
                #repeat to T x B x H x W x 1
                repeated_distance_matrix = reshaped_distance_matrix.repeat(
                    view.size()[0], view.size()[1],1,1,1)

                #we want T x BHW x 1
                repeated_distance_matrix = repeated_distance_matrix.flatten(1,3)
                conc_linear_input = torch.cat([input_feature_1, repeated_distance_matrix], dim = 2)

                linear_output = self.adapt_linear(conc_linear_input)

                #unflatten T x BHW x 1 -> T x B x H x W x 1
                linear_output = linear_output.unflatten(1, (view.size()[1], view.size()[3], view.size()[4]))
                
                #permute T x B x H x W x 1 to T x B x 1 x H x W
                linear_output = linear_output.permute(0,1,4,2,3)

                #TAKE SPARSITY LOSS OVER LINEAR OUTPUTS
                #reshape sparsity matrix 1,1,1,H,W
                reshaped_sparsity_matrix = self.sparsity_loss_weighting_matrix.reshape(1,
                        1,1, self.sparsity_loss_weighting_matrix.size()[0], self.sparsity_loss_weighting_matrix.size()[1]).to(self.device)

                #CONSIDER CHANGING TO RELU
                out_gate = self.adapt_activation(linear_output)

                #sparsity loss on linear outputs
                if self.cfg.ADAPTATION_SPARSITY_LOSS_TYPE == 'L1':
                    sp_loss_mat = torch.abs(linear_output) * reshaped_sparsity_matrix
                    sparsity_loss = torch.mean(sp_loss_mat)
                elif self.cfg.ADAPTATION_SPARSITY_LOSS_TYPE == 'olshausen':
                    sp_loss_mat = olshausen_exp_loss(linear_output) * reshaped_sparsity_matrix
                    sparsity_loss = torch.mean(sp_loss_mat)
                    print("olshausen sparsity loss is {}".format(sparsity_loss))
                elif self.cfg.ADAPTATION_SPARSITY_LOSS_TYPE == 'olshausen_v2_loss':
                    sp_loss_mat = olshausen_v2_loss(linear_output) * reshaped_sparsity_matrix
                    sparsity_loss = torch.mean(sp_loss_mat)
                elif self.cfg.ADAPTATION_SPARSITY_LOSS_TYPE == 'olshausen_v3_loss':
                    sp_loss_mat = olshausen_v3_loss(linear_output) * reshaped_sparsity_matrix
                    sparsity_loss = torch.mean(sp_loss_mat)
                    print("olshausen sparsity loss is {}".format(sparsity_loss))
                elif self.cfg.ADAPTATION_SPARSITY_LOSS_TYPE == 'modified_l1':
                    sp_loss_mat = modified_l1(out_gate) * reshaped_sparsity_matrix
                    sparsity_loss = torch.mean(sp_loss_mat)
                
                else:
                    raise Exception("sparsity loss not recognized")
                
                """
                self.save_image_as_matrix(self.sparsity_tracking_dir + "sparsity_mat_resized.png",
                        reshaped_sparsity_matrix[0,0,0,:,:].cpu().numpy())
                
                self.save_image_as_matrix(self.sparsity_tracking_dir + "sparsity_loss.png",
                        torch.mean(sp_loss_mat, dim = (0,1,2)).detach().cpu().numpy())
                """

                val = np.random.random()
                #sparsity tracking call
                if val < self.cfg.SPARSITY_TRACKING_FREQUENCY:

                    self.sparsity_tracking(.2, .8, linear_output, self.cfg.SPARSITY_IMAGE_TRACKING_FREQUENCY,
                            sparsity_loss = sp_loss_mat, sparsity_mat = reshaped_sparsity_matrix)

                adapt_out_view = view * out_gate
                assert(adapt_out_view.size() == view.size())


            elif self.cfg.ADAPTATION_TYPE == 'hardcoded_v1':
                #input size T x B x C x H x W
                T = view.size()[0]

                previous_4_mean = torch.zeros(view.size(), device = self.device)

                previous_4_mean[0,:,:,:,:] = .5
                previous_4_mean[1,:,:,:,:] = torch.mean(view[0:1,:,:,:,:], dim = 0)
                previous_4_mean[2,:,:,:,:] = torch.mean(view[0:2,:,:,:,:], dim = 0)
                previous_4_mean[3,:,:,:,:] = torch.mean(view[0:3,:,:,:,:], dim = 0)

                #want to take the mean of previous 4 timesteps for each pixel
                for t in range(4, T):
                    previous_4_mean[t,:,:,:,:] = torch.mean(view[t-4: t,:,:,:,:], dim = 0)

                #T x B x C x H x W
                input_feature_1 = torch.abs(previous_4_mean - view)
                
                spnp = np.mean(input_feature_1.cpu().numpy(), axis = (0,1,2))
                savedir = self.sparsity_tracking_dir + 'input_feature_1.png'
                plt.imsave(savedir, spnp)

                input_feature_2 = torch.mean(input_feature_1, dim = 2).reshape(input_feature_1.size()[0],
                        input_feature_1.size()[1], 1, input_feature_1.size()[3], 
                        input_feature_1.size()[4])

                spnp = np.mean(input_feature_2.cpu().numpy(), axis = (0,1,2))
                savedir = self.sparsity_tracking_dir + 'input_feature_2.png'
                plt.imsave(savedir, spnp)


                #gate
                out_gate = torch.zeros(input_feature_2.size(), device = self.device)

                #reshape sparsity matrix 1,1,1,H,W
                reshaped_sparsity_matrix = self.sparsity_loss_weighting_matrix.reshape(1,
                        1,1, self.sparsity_loss_weighting_matrix.size()[0], self.sparsity_loss_weighting_matrix.size()[1])
                
                reshaped_sparsity_matrix += .0001

                #periphery more likely to not be sent as a function of distance
                #interior less
                div_input_feature_2 = input_feature_2 / reshaped_sparsity_matrix

                spnp = np.mean(div_input_feature_2.cpu().numpy(), axis = (0,1,2))
                savedir = self.sparsity_tracking_dir + 'div_input_feature_2.png'
                plt.imsave(savedir, spnp)

                out_gate[(div_input_feature_2 > self.cfg.ADAPTATION_HARDCODED_DIFFERENCE_THRESHOLD)] = 1

                for pixel in self.do_not_change_pixel_list:
                    h,w = pixel
                    out_gate[:,:,:,h,w] = 1

                val = np.random.random()

                #sparsity tracking call
                if val < self.cfg.SPARSITY_TRACKING_FREQUENCY:
                    self.sparsity_tracking(0, 1, out_gate, self.cfg.SPARSITY_IMAGE_TRACKING_FREQUENCY)

                adapt_out_view = view * out_gate
                assert(adapt_out_view.size() == view.size())

                sparsity_loss = 0

            else:
                raise Exception("adaptation type not recognized")
            
            cnn_view = self.cnn_sensory(adapt_out_view.flatten(0, 1)).unflatten(
                0, (view.size(0), view.size(1))) # T x B x C x H x W

        else:
            #first sequential cnn layer
            cnn_view = self.cnn_sensory(view.flatten(0, 1)).unflatten(
                0, (view.size(0), view.size(1))) # T x B x C x H x W
            sparsity_loss = 0


        """
        if self.adaptation is not None:
            view = self.adaptation(view)
        cnn_view = self.cnn_sensory(view.flatten(0, 1)).unflatten(
            0, (view.size(0), view.size(1)) # T x B x C x H x W
        )
        """
        flat_cnn_view = cnn_view.flatten(2, -1) # T x B x H
        memory_input = self.sensory_to_memory(flat_cnn_view) # T x B x Hidden
        if self.cfg.INCLUDE_PROPRIO:
            saccade_focus = self._transform_proprio(saccade_focus)
            saccade_sense = saccade_focus.expand(
                -1, view.size(1), -1
            ) # T x B x 2
            memory_input = torch.cat([memory_input, saccade_sense], dim=-1)
        rnn_view, hidden_state = self.rnn(memory_input, hidden_state) # T x B x H

        return cnn_view, memory_input, rnn_view, hidden_state, sparsity_loss

    def get_views(self, image, saccades):
        # images: T x B x C x H x W
        # saccades: T x B x 2 -- !
        assert image.size(1) == 1 or saccades.size(1) == 1
        all_views = [] # raw and noised.
        w_span, h_span = self.cfg.FOV_WIDTH // 2, self.cfg.FOV_HEIGHT // 2

        # 1a. Window selection.
        # TODO Joel this can probably be vectorized
        # If anyone else knows how to do this please go ahead + cc: Joel.
        padded_image = F.pad(image, (w_span, w_span, h_span, h_span))
        for saccade in saccades.flatten(0, 1):
            w_span, h_span = self.cfg.FOV_WIDTH // 2, self.cfg.FOV_HEIGHT // 2
            view = padded_image[
                ...,
                saccade[0]: saccade[0] + 2 * w_span, # centered around saccade + span
                saccade[1]: saccade[1] + 2 * h_span, # to account for padding
            ]
            all_views.append(view)
        all_views = torch.stack(all_views, 0) # T x B x C x H x W
        if saccades.size(1) > 1:
            all_views = all_views.squeeze(1).unflatten(0, saccades.size()[:2])

        noise = torch.randn(all_views.size(), device=self.device) * self.cfg.FOV_FALLOFF
        noised_views = all_views + noise * self.view_mask
        if self.cfg.CLAMP_FOV:
            noised_views = torch.clamp(noised_views, -1, 1) # this bound is pretty aribtrary

        return all_views, noised_views

    def get_proprioception(self, image, saccades):
        if self.cfg.PROPRIOCEPTION_DELTA:
            proprioception = (saccades[1:] - saccades[:-1]).float()
            return torch.cat([
                torch.zeros((1,2), dtype=torch.float, device=self.device), proprioception
            ], dim=0)
        else:
            if self.cfg.FOURIER_PROPRIO or self.cfg.POLAR_PROPRIO:
                return saccades
            else:
                return saccades.float() / torch.tensor(image.size()[-2:], device=self.device).float() - 0.5 # 0 center

    def saccade_image(
        self,
        image,
        length=50,
        mode=None,
        initial_state: Optional[torch.tensor]=None,
    ):
        r"""
            General purpose saccading saccading sequence.
            Returns will vary based on saccading mode.
            Args:
                image: [B x C x H x W] Image batch to saccade over
                length: length of saccade
                mode: saccading mode (see `_generate_saccades`)
                initial_state: [B x H] Initial state for memory
            Returns:
                loss
                views: [T x B x C x H x W] (model observations)
                patches: [T x B x C x H x W] (model percepts).
                hidden_state: [B x H] (model memory at conclusion)
        """
        self.view_mask = self.view_mask.to(self.device) # ! weird, we can't set it in __init__()
        if self.proprio_grid is None:
            self._gen_proprio(image)
        if initial_state is None:
            hidden_state = torch.zeros((1, image.size(0), self.cfg.HIDDEN_SIZE), device=self.device)
        else:
            hidden_state = initial_state

        # Generate observations
        saccades = self._generate_saccades(image, length=length, mode=mode)
        all_views, noised_views = self.get_views(image, saccades)
        proprioception = self.get_proprioception(image, saccades)

        if all_views.size(1) != hidden_state.size(1):
            hidden_state = hidden_state.repeat(1, all_views.size(1), 1) # force batch

        # Forward and calculate loss
        cnn_view, memory_input, rnn_view, hidden_state, sparsity_loss = self(noised_views, proprioception, hidden_state)
        losses = []
        all_patches = []
        supervisory_view = noised_views if self.cfg.NOISED_SIGNAL else all_views
        for objective in self.cfg.OBJECTIVES:
            if objective == 'predictive_patch':
                all_patches = self._predict_at_location(rnn_view[:-1], saccades[1:], mode='patch')
                loss = self._evaluate_image_pred(supervisory_view[1:], all_patches)
            elif objective == 'adversarial_patch':
                # Assumes all patches is already defined
                if len(all_patches) == 0:
                    all_patches = self._predict_at_location(rnn_view[:-1], saccades[1:], mode='patch')
                fake_patches = all_patches # T B C H W
                real_disc = self.patch_contrast(cnn_view.flatten(2, -1)) # unnoised # Note the CNN must process unnoised and noised inputs, somehow
                fake_cnn_view = self.cnn_sensory(all_patches.flatten(0, 1)).unflatten(
                    0, (all_patches.size(0), all_patches.size(1)) # T x B x C x H x W
                )
                fake_disc = self.patch_contrast(fake_cnn_view.flatten(2, -1)) # T x B x H
                loss = F.binary_cross_entropy_with_logits(real_disc, torch.ones_like(real_disc)) + \
                    F.binary_cross_entropy_with_logits(fake_disc, torch.zeros_like(fake_disc))
                # TODO maybe need to subsample or make this less aggressive or whatever
                # whoa whoa whoa this is all wrong.
                # we need to make this **adversarial** dude. you're making it **cooperative**.
            elif objective == 'autoencode': # Legacy, debug objective
                if self.cfg.REACTIVE:
                    all_patches = self.cnn_predictive(cnn_view.flatten(0, 1)).unflatten(0, (cnn_view.size(0), cnn_view.size(1)))
                else:
                    all_patches = self._predict_at_location(rnn_view, saccades) # Model must reproduce exactly what it saw.
                loss = self._evaluate_image_pred(supervisory_view, all_patches)
            elif objective == 'predictive':
                # Each module predicts its input (implying autoencoding for CNN)
                all_patches = self.cnn_predictive(
                    cnn_view.flatten(0, 1)
                ).unflatten(0, (cnn_view.size(0), cnn_view.size(1)))
                loss_cnn = self._evaluate_image_pred(supervisory_view, all_patches)
                cnn_predictions = self._predict_at_location(rnn_view[:-1], saccades[1:], mode='percept')
                loss_rnn = self.mse(cnn_view[1:], cnn_predictions)
                loss = loss_cnn + loss_rnn
            elif objective == 'contrast':
                raise NotImplementedError
                # minibatch contrast of memory input T B H
                # NUM_PAIRS = 8
                # # TODO index random batches...
                # time_pairs = torch.randint(0, memory_input.size(0), (NUM_PAIRS * 2)) #
                # batch_negatives = torch.randint(0, memory_input.size(1), (NUM_PAIRS, 2))
                # batch_positives = torch.randint(0, memory_input.size(1), (NUM_PAIRS, 1)).expand(NUM_PAIRS, 2)
                # positives = torch.gather(
                #     memory_input.flatten(0, 1),
                #     dim=0,
                #     index=time_pairs.expand_as(memory_input.size())
                # ).reshape(NUM_PAIRS, 2 * memory_input.size(0), memory_input.size(1), memory_input.size(2)) # P x 2T x H
                # #
                # negatives = torch.gather(
                #     positives.view(t * n, -1),
                #     dim=0,
                #     index=negative_inds.view(t * n, 1).expand(-1, positives.size(-1)),
                # ).view(t, n, -1)
            elif objective == 'random':
                raise NotImplementedError # TODO Joel
            else:
                raise Exception
            

            losses.append(loss)


        if np.random.random() < .01:
            print("sparsity loss is {} loss is {}".format(sparsity_loss, loss))
            #    print("".format(loss))
        
        loss = torch.stack(losses).sum() + sparsity_loss # no tradeoff terms

        # TODO how do we decide what patches to return?
        # Currently it'll return the last objective's patches
        return loss, all_views, noised_views, all_patches, hidden_state

    # Pytorch lightning API below
    # TODO systematic prediction to get a "full image" percept.
    # Problem is that we don't train in any systematic matter
    # So grid-based reconstruction is probably not exactly accurate.

    @torch.no_grad()
    def predict(
        self,
        image,
        saccade_first=True,
        mode='autoencode'
    ):
        r"""
            Will take the image, saccade randomly over it, and then make predictions (on a different random saccade sequence).
            Args:
                image: C x H x W.
        """
        loss, all_views, noised_views, all_patches, state = self.saccade_image(image.unsqueeze(0))
        print('First saccade:', loss)
        if saccade_first: # ok... what now.
            loss, all_views, noised_views, all_patches, state = \
            self.saccade_image(image.unsqueeze(0), initial_state=state)
            print('Next saccade:', loss)

        if self.cfg.QUANTIZED_RECONSTRUCTION:
            all_patches = self._quantized_pred(all_patches)
        return all_views, noised_views, all_patches, state

    @torch.no_grad()
    def predict_with_saccades(
        self,
        image,
        saccades,
        initial_state=None,
        mode='predictive_patch'
    ):
        r"""
            Finer control
        """
        image = image.unsqueeze(0)
        if self.proprio_grid is None:
            self._gen_proprio(image)
        all_patches = []

        if initial_state is None:
            hidden_state = torch.zeros((1, image.size(0), self.cfg.HIDDEN_SIZE), device=self.device)
        else:
            hidden_state = initial_state
        # Generate observations
        all_views, noised_views = self.get_views(image, saccades)
        proprioception = self.get_proprioception(image, saccades)

        cnn_view, flat_cnn_view, rnn_view, hidden_state, sparsity_loss = self(noised_views, proprioception, hidden_state)
        if mode == 'predictive_patch':
            all_patches = self._predict_at_location(rnn_view[:-1], saccades[1:], mode='patch')
        elif mode == 'autoencode': # Legacy, debug objective
            all_patches = self.cnn_predictive(
                cnn_view.flatten(0, 1)
            ).unflatten(0, (cnn_view.size(0), cnn_view.size(1)))
        return all_views, noised_views, all_patches, hidden_state

    @torch.no_grad()
    def predict_grid(
        self,
        image,
        saccades, # preliminary saccade
        initial_state=None,
    ):
        r"""
            Finer control
        """
        image = image.unsqueeze(0)
        all_patches = []

        if initial_state is None:
            hidden_state = torch.zeros((1, image.size(0), self.cfg.HIDDEN_SIZE), device=self.device)
        else:
            hidden_state = initial_state
        # Generate observations
        all_views, noised_views = self.get_views(image, saccades)
        proprioception = self.get_proprioception(image, saccades)

        cnn_view, flat_cnn_view, rnn_view, hidden_state, sparsity_loss = self(noised_views, proprioception, hidden_state)





        # Use the final rnn state to make predictions about the whole grid.
        # grid_h = np.linspace(0, image.size(-2), 4)
        # grid_w = np.linspace(0, image.size(-1), 4)
        all_patches = self._predict_at_location(rnn_view[:-1], saccades[1:], mode='patch')
        return all_views, noised_views, all_patches, hidden_state

    def training_step(self, batch, batch_idx):
        # batch - B x H x W
        loss, *_ = self.saccade_image(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, *_ = self.saccade_image(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, *_ = self.saccade_image(batch)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Reduce LR on plateau as a reasonable default
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)
        # optimizer_generator = optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)
        # optimizer_discriminator = optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)
        return {
            'optimizer': optimizer,
            'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50),
            'monitor': 'val_loss'
        }


def upsample_conv(c_in, c_out, scale_factor, k_size=3, stride=1, pad=1, bn=True):
    return nn.Sequential(
        nn.Upsample(scale_factor=scale_factor),
        nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False),
        nn.BatchNorm2d(c_out)
    )


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)