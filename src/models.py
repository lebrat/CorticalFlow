import torch
import torch.nn as nn
import torch.nn.functional as F


def save_checkpoint(deform_train, ite, model, optimizer, lr_scheduler, best_val_loss, ckp_file):
    torch.save({'deform_train': deform_train,
                'iteration': ite,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_schedule_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                'best_val_loss': best_val_loss}, 
                ckp_file)

def load_checkpoint(ckp_file, model=None, optimizer=None, lr_scheduler=None):
    chk_dict = torch.load(ckp_file)
    deform_train, iteration, best_val_loss = chk_dict['deform_train'], chk_dict['iteration'], chk_dict['best_val_loss']
    
    if model is not None:
        model.load_state_dict(chk_dict['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(chk_dict['optimizer_state_dict'])
    if chk_dict['lr_schedule_state_dict'] is not None and lr_scheduler is not None:
        lr_scheduler.load_state_dict(chk_dict['lr_schedule_state_dict'])
    
    return deform_train, iteration, best_val_loss
            

class CorticalFlow(nn.Module):
    def __init__(self, combine_flow, nb_features, inte_method, inte_steps):
        super(CorticalFlow, self).__init__()
        self.num_deforms = len(nb_features)
        self.combine_flow =  combine_flow
        
        self.deform_blocks = nn.ModuleList()
        in_channels = 1
        for d_idx in range(self.num_deforms):
            self.deform_blocks.append(DeformationBlock(nb_features[d_idx], in_channels, inte_method, inte_steps))
            if self.combine_flow: in_channels += 3


    def forward(self, input, input_affine, template_verts, use_deforms):        
        # check data        
        if input.ndim == 4: input = input.unsqueeze(1)
        assert input.ndim == 5
        batch_size, channels, width, height, depth = input.shape
        assert input_affine.shape == (batch_size, 4, 4) 
        assert template_verts.ndim == 3, template_verts.shape[[0,-1]] == (batch_size, 3) 
        assert all(ud in range(self.num_deforms) for ud in use_deforms)

        scale_list, flow_field_list, flow_field_int_list, pred_verts_list = [], [], [], []
        for d_idx in use_deforms:
            
            # compute deformation
            scale, flow_field, flow_field_int, pred_verts = self.deform_blocks[d_idx](input, input_affine, template_verts)
            
            # join results
            scale_list += scale; flow_field_list += flow_field; 
            flow_field_int_list += flow_field_int; pred_verts_list += pred_verts;

            # update inputs
            if self.combine_flow: input = torch.cat([input, flow_field_list[-1]], dim=1)            
            template_verts = pred_verts_list[-1]

        return scale_list, flow_field_list, flow_field_int_list, pred_verts_list


class DeformationBlock(nn.Module):
    def __init__(self, nb_features, ch_first_layer, inte_method, inte_steps):
        super(DeformationBlock, self).__init__()

        inshape = [96, 192, 160]
        self.encoder = Unet(inshape, nb_features=nb_features, nb_levels=None, feat_mult=None, ch_first_layer=ch_first_layer)
               
        # configure unet to flow field layer
        self.flow = nn.Conv3d(self.encoder.dec_nf[-1], 3, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(torch.distributions.normal.Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # diffeomorphic deformation module (DMD)
        self.deformer = DiffeoMeshDeformer(inte_method, inte_steps)


    def forward(self, input, input_affine, template_verts):
        # read in data                
        if input.ndim == 4: input = input.unsqueeze(1)
        assert input.ndim == 5
        batch_size, channels, width, height, depth = input.shape

        # propagate UNET
        x = self.encoder(input)

        # transform into flow field
        flow_field = self.flow(x)
        
        # diffeomorphic wrapper
        pred, flow_field_int = self.deformer(template_verts, input_affine, flow_field)
        
        return [1.0], [flow_field], [flow_field_int], [pred]


class DiffeoMeshDeformer(nn.Module):
    def __init__(self, inte_method, inte_steps):
        super(DiffeoMeshDeformer, self).__init__()

        # configure optional integration layer for diffeomorphic warp        
        self.inte_method, self.inte_steps = inte_method, inte_steps
        if self.inte_method == 'NeurIPS':
            self.integrate = IntegratePointWiseNeurips(self.inte_steps)
        elif self.inte_method == 'Euler':
            self.integrate = IntegratePointWiseEuler(self.inte_steps)
        elif self.inte_method == 'RK4':
            self.integrate = IntegratePointWiseRK4(self.inte_steps)                        
        elif self.inte_method == None:      
            self.integrate, self.point_pool = None, PointPooling()            
        else:
            raise ValueError("integration method {} is not supported".format(self.inte_method))        


    def forward(self, verts, affine, flow_field):
        
        # map world coordinates to mri voxels              
        pred = torch.cat([verts, torch.ones((verts.shape[0], verts.shape[1], 1), device=verts.device)], dim=-1)        
        pred = torch.matmul(affine, pred.transpose(2, 1)).transpose(2, 1)[:, :, :-1]

        # integrate to produce diffeomorphic warp and move vertices          
        if self.inte_method in ['NeurIPS', 'Euler', 'RK4']:
            aux_pred = self.integrate(flow_field, pred)    
            flow_field_int = aux_pred - pred; pred =  1.0 * aux_pred;
            flow_field_int = flow_field_int.permute(0, 2, 1).unsqueeze(dim=-1).unsqueeze(dim=-1)           
        else:
            assert self.inte_method == None 
            pred = pred + self.point_pool(flow_field, pred).transpose(2, 1)        
            flow_field_int = 1.0 * flow_field    

        # map mri voxels to world cordinates
        pred = torch.cat([pred, torch.ones((pred.shape[0], pred.shape[1], 1), device=pred.device)], dim=-1)        
        pred = torch.matmul(torch.inverse(affine), pred.transpose(2, 1)).transpose(2, 1)[:, :, :-1]        

        return pred, flow_field_int


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1, use_deconv=False, ch_first_layer=1):
        super(Unet, self).__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        self.use_deconv =  use_deconv
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = [
                [16, 32, 32, 32],             # encoder
                [32, 32, 32, 32, 32, 16, 16]  # decoder
            ]

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = ch_first_layer
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf          
            if self.use_deconv:
                self.uparm.append(nn.Sequential(ConvBlock(ndims, channels, nf, stride=1), nn.ConvTranspose3d(nf, nf, 2, stride=2)))            
            else:
                self.uparm.append(nn.Sequential(ConvBlock(ndims, channels, nf, stride=1), nn.Upsample(scale_factor=2, mode='nearest'))) 
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += ch_first_layer
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf
 
    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            # x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x



#### CUSTOM LAYERS

class IntegratePointWiseNeurips(nn.Module):
    """ NEURIPS Scaling version"""
    def __init__(self, num_steps):
        super(IntegratePointWiseNeurips, self).__init__()
        self.interpolator = PointPooling()
        self.num_steps = num_steps    
    
    def forward(self, flow, x):            
        scale = 1.0/(2**self.num_steps)
        flow = flow*scale
        for i in range(self.num_steps):
            x = x + self.interpolator(flow,x).transpose(2, 1)

        return x


class IntegratePointWiseEuler(nn.Module):
    """ Correct Scaling version"""
    def __init__(self, num_steps):
        super(IntegratePointWiseEuler, self).__init__()
        self.interpolator = PointPooling()
        self.num_steps = num_steps
    
    def forward(self, flow, x):            
        scale = 1.0 / self.num_steps
        flow = flow * scale
        for i in range(self.num_steps):
            x = x + self.interpolator(flow,x).transpose(2, 1)

        return x


class IntegratePointWiseRK4(nn.Module):
    def __init__(self, num_steps):
        super(IntegratePointWiseRK4, self).__init__()
        self.interpolator = PointPooling()
        self.num_steps = num_steps
        self.weights = [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]
        self.coefs = [0.0, 0.5, 0.5, 1.0]        
    
    def forward(self, flow, x):                 
        flow = flow * (1.0 / self.num_steps)
        for i in range(self.num_steps):
            prev_k, step = 0.0, 0.0            
            for j in range(len(self.weights)):                
                prev_k = self.interpolator(flow, x + self.coefs[j] * prev_k).transpose(2, 1)
                step += self.weights[j] * prev_k
            x = x + step

        return x

    
class PointPooling(nn.Module):
    """
    Local pooling operation.
    """
    def __init__(self, interpolation='bilinear'):
        super(PointPooling, self).__init__()
        self.interp_mode = interpolation

    def forward(self, x, v):          
        batch, num_points, num_feats = x.size(0), v.size(1), x.size(1)
        norm_v = torch.tensor([[[x.size(2)-1, x.size(3)-1, x.size(4)-1]]], device='cuda').float()
        norm_v = 2 * (v/norm_v) - 1
        grid = norm_v.unsqueeze(dim=-2).unsqueeze(dim=-2).flip(dims=(-1,))

        out = F.grid_sample(x, grid, mode=self.interp_mode, padding_mode='border', align_corners=True)
        out = out.squeeze().view(batch, num_feats, num_points)
        return out

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class Debug(nn.Module):
    def __init__(self, func):
        super(Debug, self).__init__()
        self.func = func

    def forward(self, x):
        self.func(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)