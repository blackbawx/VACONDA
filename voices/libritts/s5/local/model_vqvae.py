import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)

from models import *
from layers import *
from util import *
from model import *


# https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/upsample.py
class UpsampleNetwork(nn.Module):
    def __init__(self, upsample_scales, upsample_activation="none",
                 upsample_activation_params={}, mode="nearest",
                 freq_axis_kernel_size=1, cin_pad=0, cin_channels=80):
        super(UpsampleNetwork, self).__init__()
        self.up_layers = nn.ModuleList()
        total_scale = np.prod(upsample_scales)
        self.indent = cin_pad * total_scale
        for scale in upsample_scales:
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            k_size = (freq_axis_kernel_size, scale * 2 + 1)
            padding = (freq_axis_padding, scale)
            stretch = Stretch2d(scale, 1, mode)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / np.prod(k_size))
            conv = nn.utils.weight_norm(conv)
            self.up_layers.append(stretch)
            self.up_layers.append(conv)
            if upsample_activation != "none":
                nonlinear = _get_activation(upsample_activation)
                self.up_layers.append(nonlinear(**upsample_activation_params))

    def forward(self, c):
   
        c = c.transpose(1,2)

        # B x 1 x C x T
        c = c.unsqueeze(1)
        for f in self.up_layers:
            c = f(c)
        # B x C x T
        c = c.squeeze(1)

        if self.indent > 0:
            c = c[:, :, self.indent:-self.indent]
        return c.transpose(1,2)

class Stretch2d(nn.Module):
    def __init__(self, x_scale, y_scale, mode="nearest"):
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)


def _get_activation(upsample_activation):
    nonlinear = getattr(nn, upsample_activation)
    return nonlinear


# https://github.com/mkotha/WaveRNN/blob/master/layers/downsampling_encoder.py 
class DownsamplingEncoderStrict(nn.Module): 
    """ 
        Input: (N, samples_i) numeric tensor 
        Output: (N, samples_o, channels) numeric tensor 
    """ 
    def __init__(self, channels, layer_specs, input_dim = 1, use_batchnorm=0): 
        super().__init__() 
 
        self.convs_wide = nn.ModuleList() 
        self.convs_1x1 = nn.ModuleList() 
        self.layer_specs = layer_specs 
        prev_channels = input_dim
        total_scale = 1 
        pad_left = 0 
        self.skips = [] 
        for stride, ksz, dilation_factor in layer_specs: 
            conv_wide = nn.Conv1d(prev_channels, 2 * channels, ksz, stride=stride, dilation=dilation_factor, padding=int((ksz-1)/2)) 
            wsize = 2.967 / math.sqrt(ksz * prev_channels) 
            conv_wide.weight.data.uniform_(-wsize, wsize) 
            conv_wide.bias.data.zero_() 
            self.convs_wide.append(conv_wide) 
 
            conv_1x1 = nn.Conv1d(channels, channels, 1) 
            conv_1x1.bias.data.zero_() 
            self.convs_1x1.append(conv_1x1) 
 
            prev_channels = channels 
            skip = (ksz - stride) * dilation_factor 
            pad_left += total_scale * skip 
            self.skips.append(skip) 
            total_scale *= stride 
        self.pad_left = pad_left 
        self.total_scale = total_scale 
        self.final_conv_0 = nn.Conv1d(channels, channels, 1) 
        self.final_conv_0.bias.data.zero_() 
        self.final_conv_1 = nn.Conv1d(channels, channels, 1) 
        self.batch_norm = nn.BatchNorm1d(channels, momentum=0.9) 
        self.use_batchnorm = use_batchnorm

    def forward(self, samples):
        x = samples.transpose(1,2) #.unsqueeze(1)
        #print("Shape of input: ", x.shape)
        for i, stuff in enumerate(zip(self.convs_wide, self.convs_1x1, self.layer_specs, self.skips)):
            conv_wide, conv_1x1, layer_spec, skip = stuff
            stride, ksz, dilation_factor = layer_spec
            #print(i, "Stride, ksz, DF and shape of input: ", stride, ksz, dilation_factor, x.shape)
            x1 = conv_wide(x)
            x1_a, x1_b = x1.split(x1.size(1) // 2, dim=1)
            x2 = torch.tanh(x1_a) * torch.sigmoid(x1_b)
            x3 = conv_1x1(x2)
            #if i == 0:
            #    x = x3
            #else:
            #    x = x3 + x[:, :, skip:skip+x3.size(2)*stride].view(x.size(0), x3.size(1), x3.size(2), -1)[:, :, :, -1]
            x = x3
        x = self.final_conv_1(F.relu(self.final_conv_0(x)))
        #print("Shape of output: ", x.shape)
        if self.use_batchnorm:
           return self.batch_norm(x).transpose(1, 2)
        return x.transpose(1,2)

class MelLSTM(nn.Module):

    def __init__(self, input_dim = 256, mel_dim = 80):
        super(MelLSTM, self).__init__()

        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.joint_encoder = nn.LSTM(input_dim + mel_dim, 128, batch_first=True)

        self.hidden2hidden = nn.Linear(128, 128)
        self.hidden2logits = nn.Linear(128, 80)
        self.drop = nn.Dropout(0.3)
        self.leakyrelu = nn.LeakyReLU(0.1)


    def forward(self, mels, x):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        #print("Shapes of mels and x: ", mels.shape, x.shape)

        mels = mels[:,:-1,:]
        inputs = x[:, :-1, :]
        #print("Shapes of mels and x: ", mels.shape, x.shape)

        melsNinputs = torch.cat([mels, inputs], dim=-1)

        hidden,_ = self.joint_encoder(melsNinputs)
        hidden = torch.tanh(self.hidden2hidden(hidden))
        logits = self.hidden2logits(hidden)

        return logits, x[:,1:]


    def forward_generate(self, mels):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        T = mels.size(1)

        input_float = torch.zeros(mels.shape[0], 80).cuda() #+ 0.0034
        output = []
        hidden = None

        for i in range(T):

           # Concatenate mel and coarse_float
           m = mels[:, i,:]
           #print("Shape of m and input_float: ", m.shape, input_float.shape)
           inp = torch.cat([m , input_float.cuda()], dim=-1).unsqueeze(1)

           # Get coarse and fine logits
           mels_encoded, hidden = self.joint_encoder(inp, hidden)
           hidden_ = torch.tanh(self.hidden2hidden(mels_encoded))
           logits = self.hidden2logits(hidden_)

           if i%10000 == 1:
              print("Processed timesteps ", i, T)

           # Generate sample at current time step
           output.append(logits)

           # Estimate the input for next time step
           input_float = logits
           input_float = input_float.squeeze(1)

        #return np.array(output)
        output = torch.stack(output, dim=0).squeeze(1)
        print("Shape of output: ", output.shape)
        return output #.cpu().numpy()



class quantizer_kotha_arff(nn.Module):
    """
        Input: (B, T, n_channels, vec_len) numeric tensor n_channels == 1 usually
        Output: (B, T, n_channels, vec_len) numeric tensor
    """
    def __init__(self, n_channels, n_classes, vec_len, normalize=False, scale=None, assistant=None):
        super().__init__()
        if normalize:
            target_scale = scale if scale is not None else  0.06
            self.embedding_scale = target_scale
            self.normalize_scale = target_scale
        else:
            self.embedding_scale = 1e-3 #1e-3
            self.normalize_scale = None
        self.embedding0_2classes = nn.Parameter(torch.randn(n_channels, 2, vec_len, requires_grad=True) * self.embedding_scale)
        self.embedding0_3classes = nn.Parameter(torch.randn(n_channels, 3, vec_len, requires_grad=True) * self.embedding_scale)
        self.embedding0_4classes = nn.Parameter(torch.randn(n_channels, 4, vec_len, requires_grad=True) * self.embedding_scale)
        self.embedding0_nclasses = nn.Parameter(torch.randn(n_channels, 16, vec_len, requires_grad=True) * self.embedding_scale)
        self.offset = torch.arange(n_channels).cuda() * n_classes
        # self.offset: (n_channels) long tensor
        self.n_classes = n_classes
        self.after_update()
        self.plot_histogram = 0
        self.assistant = assistant

    def forward(self, x0, chunk_size=512):
        fig = None
        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding_2classes = target_norm * self.embedding0_2classes / self.embedding0_2classes.norm(dim=2, keepdim=True)
            embedding_3classes = target_norm * self.embedding0_3classes / self.embedding0_3classes.norm(dim=2, keepdim=True)
            embedding_4classes = target_norm * self.embedding0_4classes / self.embedding0_4classes.norm(dim=2, keepdim=True)
            embedding_nclasses = target_norm * self.embedding0_nclasses / self.embedding0_nclasses.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding_2classes = self.embedding0_2classes
            embedding_3classes = self.embedding0_3classes
            embedding_4classes = self.embedding0_4classes
            embedding_nclasses = self.embedding0_nclasses


        #logger.log(f'std[x] = {x.std()}')
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor
        #print("Shape of x1 and embedding: ", x1.shape, embedding.shape)

        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks_2classes = []
        index_chunks_3classes = []
        index_chunks_4classes = []
        index_chunks_nclasses = []
        for x1_chunk in x1.split(chunk_size, dim=0):
            #print("Shapes of x1_chunk, embedding_2classes, embedding_3classes and embedding_4classes: ", x1_chunk[:,:,:,:63].shape, embedding_2classes.shape, embedding_3classes.shape, embedding_4classes.shape)
            index_chunks_2classes.append((x1_chunk[:, :,:, 0:64] - embedding_2classes).norm(dim=3).argmin(dim=2))
            index_chunks_3classes.append((x1_chunk[:, :,:,64:128] - embedding_3classes).norm(dim=3).argmin(dim=2))
            index_chunks_4classes.append((x1_chunk[:,:,:,128:192] - embedding_4classes).norm(dim=3).argmin(dim=2))
            index_chunks_nclasses.append((x1_chunk[:,:,:,192:256] - embedding_nclasses).norm(dim=3).argmin(dim=2))

        index_2classes = torch.cat(index_chunks_2classes, dim=0)
        index_3classes = torch.cat(index_chunks_3classes, dim=0)
        index_4classes = torch.cat(index_chunks_4classes, dim=0)
        index_nclasses = torch.cat(index_chunks_nclasses, dim=0)

        # index: (N*samples, n_channels) long tensor
           
        hist_2classes = index_2classes.float().cpu().histc(bins=2, min=-0.5, max=1.5)
        hist_3classes = index_3classes.float().cpu().histc(bins=3, min=-0.5, max=2.5)
        hist_4classes = index_4classes.float().cpu().histc(bins=4, min=-0.5, max=3.5)
        hist_nclasses = index_nclasses.float().cpu().histc(bins=64, min=-0.5, max=3.5)

        if self.plot_histogram:  
               assert self.assistant is not None

               hists = hist_2classes.cpu().numpy()
               fig = plt.figure() 
               # https://stackoverflow.com/questions/51473993/plot-an-histogram-with-y-axis-as-percentage-using-funcformatter
               plt.hist(hists, weights=np.ones(2) / 2) 
               plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
               self.assistant.log_image('latent_histograms_2classes', fig)

               hists = hist_3classes.cpu().numpy()
               fig = plt.figure() 
               # https://stackoverflow.com/questions/51473993/plot-an-histogram-with-y-axis-as-percentage-using-funcformatter
               plt.hist(hists, weights=np.ones(3) / 3) 
               plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
               self.assistant.log_image('latent_histograms_3classes', fig) 
               plt.close()

               hists = hist_4classes.cpu().numpy()
               fig = plt.figure() 
               # https://stackoverflow.com/questions/51473993/plot-an-histogram-with-y-axis-as-percentage-using-funcformatter
               plt.hist(hists, weights=np.ones(4) / 4) 
               plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
               self.assistant.log_image('latent_histograms_4classes', fig) 
               plt.close()

               hists = hist_nclasses.cpu().numpy()
               fig = plt.figure() 
               # https://stackoverflow.com/questions/51473993/plot-an-histogram-with-y-axis-as-percentage-using-funcformatter
               plt.hist(hists, weights=np.ones(64) / 64) 
               plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
               self.assistant.log_image('latent_histograms_nclasses', fig) 
               plt.close()

               self.plot_histogram = 0

        prob_2classes = hist_2classes.masked_select(hist_2classes > 0) / len(index_2classes)
        entropy_2classes = - (prob_2classes * prob_2classes.log()).sum().item()

        prob_3classes = hist_3classes.masked_select(hist_3classes > 0) / len(index_3classes)
        entropy_3classes = - (prob_3classes * prob_3classes.log()).sum().item()

        prob_4classes = hist_4classes.masked_select(hist_4classes > 0) / len(index_4classes)
        entropy_4classes = - (prob_4classes * prob_4classes.log()).sum().item()

        prob_nclasses = hist_nclasses.masked_select(hist_nclasses > 0) / len(index_nclasses)
        entropy_nclasses = - (prob_nclasses * prob_nclasses.log()).sum().item()

           
        index1_2classes = (index_2classes + self.offset).view(index_2classes.size(0) * index_2classes.size(1))
        index1_3classes = (index_3classes + self.offset).view(index_3classes.size(0) * index_3classes.size(1))
        index1_4classes = (index_4classes + self.offset).view(index_4classes.size(0) * index_4classes.size(1))
        index1_nclasses = (index_nclasses + self.offset).view(index_nclasses.size(0) * index_nclasses.size(1))

        # index1: (N*samples*n_channels) long tensor
        output_flat_2classes = embedding_2classes.view(-1, embedding_2classes.size(2)).index_select(dim=0, index=index1_2classes)
        output_flat_3classes = embedding_3classes.view(-1, embedding_3classes.size(2)).index_select(dim=0, index=index1_3classes)
        output_flat_4classes = embedding_4classes.view(-1, embedding_4classes.size(2)).index_select(dim=0, index=index1_4classes)
        output_flat_nclasses = embedding_nclasses.view(-1, embedding_nclasses.size(2)).index_select(dim=0, index=index1_nclasses)

        # output_flat: (N*samples*n_channels, vec_len) numeric tensor
        output_2classes = output_flat_2classes.view(x.shape[0], x.shape[1], x.shape[2], -1)
        output_3classes = output_flat_3classes.view(x.shape[0], x.shape[1], x.shape[2], -1)
        output_4classes = output_flat_4classes.view(x.shape[0], x.shape[1], x.shape[2], -1)
        output_nclasses = output_flat_nclasses.view(x.shape[0], x.shape[1], x.shape[2], -1)

        output = torch.cat([output_2classes, output_3classes, output_4classes, output_nclasses], dim=-1) 
        #print("Shape of output and x: ", output.shape, x.shape, output_2classes.shape)

        out0 = (output - x).detach() + x
        out1 = (x.detach() - output).float().norm(dim=3).pow(2)
        out2 = (x - output.detach()).float().norm(dim=3).pow(2) + (x - x0).float().norm(dim=3).pow(2)
        #logger.log(f'std[embedding0] = {self.embedding0.view(-1, embedding.size(2)).index_select(dim=0, index=index1).std()}')
        return (out0, out1, out2, entropy_2classes, entropy_3classes, entropy_4classes, entropy_nclasses)


    def after_update(self):
        if self.normalize_scale:
            with torch.no_grad():
                target_norm = self.embedding_scale * math.sqrt(self.embedding0_2classes.size(2))
                self.embedding0_2classes.mul_(target_norm / self.embedding0_2classes.norm(dim=2, keepdim=True))

                target_norm = self.embedding_scale * math.sqrt(self.embedding0_3classes.size(2))
                self.embedding0_3classes.mul_(target_norm / self.embedding0_3classes.norm(dim=2, keepdim=True))

                target_norm = self.embedding_scale * math.sqrt(self.embedding0_4classes.size(2))
                self.embedding0_4classes.mul_(target_norm / self.embedding0_4classes.norm(dim=2, keepdim=True))
            
                target_norm = self.embedding_scale * math.sqrt(self.embedding0_nclasses.size(2))
                self.embedding0_nclasses.mul_(target_norm / self.embedding0_nclasses.norm(dim=2, keepdim=True))


    def get_quantizedindices(self, x0, chunk_size=512):
        fig = None
        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding_2classes = target_norm * self.embedding0_2classes / self.embedding0_2classes.norm(dim=2, keepdim=True)
            embedding_3classes = target_norm * self.embedding0_3classes / self.embedding0_3classes.norm(dim=2, keepdim=True)
            embedding_4classes = target_norm * self.embedding0_4classes / self.embedding0_4classes.norm(dim=2, keepdim=True)
            embedding_nclasses = target_norm * self.embedding0_nclasses / self.embedding0_nclasses.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding_2classes = self.embedding0_2classes
            embedding_3classes = self.embedding0_3classes
            embedding_4classes = self.embedding0_4classes
            embedding_nclasses = self.embedding0_nclasses


        #logger.log(f'std[x] = {x.std()}')
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor
        #print("Shape of x1 and embedding: ", x1.shape, embedding.shape)
            
        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks_2classes = []
        index_chunks_3classes = []
        index_chunks_4classes = []
        index_chunks_nclasses = []
        for x1_chunk in x1.split(chunk_size, dim=0):
            #print("Shapes of x1_chunk, embedding_2classes, embedding_3classes and embedding_4classes: ", x1_chunk[:,:,:,:63].shape, embedding_2classes.shape, embedding_3classes.shape, embedding_4classes$
            index_chunks_2classes.append((x1_chunk[:, :,:, 0:64] - embedding_2classes).norm(dim=3).argmin(dim=2))
            index_chunks_3classes.append((x1_chunk[:, :,:,64:128] - embedding_3classes).norm(dim=3).argmin(dim=2))
            index_chunks_4classes.append((x1_chunk[:,:,:,128:192] - embedding_4classes).norm(dim=3).argmin(dim=2))
            index_chunks_nclasses.append((x1_chunk[:,:,:,192:256] - embedding_nclasses).norm(dim=3).argmin(dim=2))
        
        index_2classes = torch.cat(index_chunks_2classes, dim=0)
        index_3classes = torch.cat(index_chunks_3classes, dim=0)
        index_4classes = torch.cat(index_chunks_4classes, dim=0)
        index_nclasses = torch.cat(index_chunks_nclasses, dim=0)

        # index: (N*samples, n_channels) long tensor

        hist_2classes = index_2classes.float().cpu().histc(bins=2, min=-0.5, max=1.5)
        hist_3classes = index_3classes.float().cpu().histc(bins=3, min=-0.5, max=2.5)
        hist_4classes = index_4classes.float().cpu().histc(bins=4, min=-0.5, max=3.5)
        hist_nclasses = index_nclasses.float().cpu().histc(bins=64, min=-0.5, max=3.5)

        prob_2classes = hist_2classes.masked_select(hist_2classes > 0) / len(index_2classes)
        entropy_2classes = - (prob_2classes * prob_2classes.log()).sum().item()

        prob_3classes = hist_3classes.masked_select(hist_3classes > 0) / len(index_3classes)
        entropy_3classes = - (prob_3classes * prob_3classes.log()).sum().item()
        
        prob_4classes = hist_4classes.masked_select(hist_4classes > 0) / len(index_4classes)
        entropy_4classes = - (prob_4classes * prob_4classes.log()).sum().item()
        
        prob_nclasses = hist_nclasses.masked_select(hist_nclasses > 0) / len(index_nclasses)
        entropy_nclasses = - (prob_nclasses * prob_nclasses.log()).sum().item()

        index1_2classes = (index_2classes.squeeze(1) + self.offset).cpu().numpy().tolist()
        index1_3classes = (index_3classes.squeeze(1) + self.offset).cpu().numpy().tolist()
        index1_4classes = (index_4classes.squeeze(1) + self.offset).cpu().numpy().tolist()
        index1_nclasses = (index_nclasses.squeeze(1) + self.offset).cpu().numpy().tolist()

        latents_2classes = ' '.join(str(k) for k in self.deduplicate(index1_2classes))
        latents_3classes = ' '.join(str(k) for k in self.deduplicate(index1_3classes))
        latents_4classes = ' '.join(str(k) for k in self.deduplicate(index1_4classes))
        latents_nclasses = ' '.join(str(k) for k in self.deduplicate(index1_nclasses))

        print("2 Class Latents and entropy: ", latents_2classes, entropy_2classes)
        print("3 Class Latents and entropy: ", latents_3classes, entropy_3classes)
        print("4 Class Latents and entropy: ", latents_4classes, entropy_4classes)
        print("N Class Latents and entropy: ", latents_nclasses, entropy_nclasses)

    # Remove repeated entries
    def deduplicate(self, arr):
       arr_new = []
       current_element = None
       for element in arr:
          if current_element is None:
            current_element = element
            arr_new.append(element)
          elif element == current_element:
            continue
          else:
            current_element = element
            arr_new.append(element)
       return arr_new


class VACONDA(nn.Module):     
    def __init__(self, embedding_dim=256, input_dim=80, r = 4, mel_dim = 80, linear_dim = 1025, num_spk = 2, use_arff = 0, assistant = None):     
        super(VACONDA, self).__init__()  
        if use_arff:
            self.quantizer = quantizer_kotha_arff(n_channels=1, n_classes=256, vec_len=int(embedding_dim/4), normalize=True, assistant = assistant)     
        else:  
            self.quantizer = quantizer_kotha(n_channels=1, n_classes=16, vec_len=embedding_dim, normalize=True, assistant = assistant)     
        encoder_layers = [     
            #(2, 4, 1),    
            #(2, 4, 1),     
            #(2, 4, 1),     
            #(2, 4, 1),     
            (2, 4, 1),    
            #(1, 4, 1),
            (2, 4, 1),    
            ]     
        self.downsampling_encoder = DownsamplingEncoderStrict(embedding_dim, encoder_layers, input_dim=mel_dim, use_batchnorm=1)     
        #self.decoder = SpecLSTM(input_dim=embedding_dim)  
        self.embedding_fc = nn.Linear(256, 128)  
        #self.decoder.upsample_scales = [2,2]  
        #self.decoder.upsample_network = UpsampleNetwork(self.decoder.upsample_scales)  
        self.r = r 
        self.postnet = CBHG(mel_dim, K=8, projections=[256, mel_dim])  
        self.mel_dim = mel_dim
        self.last_linear = nn.Linear(mel_dim * 2, linear_dim)
        print("Outputs per step: ", r)
        #self.lid_postnet = CBHG(embedding_dim, K=8, projections=[256, embedding_dim]) 
        self.lid_lstm = nn.LSTM(embedding_dim, 128, bidirectional=True, batch_first=True)
        self.lid_fc = nn.Linear(128, 2)
        self.use_arff = use_arff
        self.decoder_mellstm = MelLSTM(input_dim = 384, mel_dim = 80)
        self.spk_embedding = nn.Embedding(num_spk, 128)
        self.spk_embedding.weight.data.normal_(0, 0.3)
        self.upsample_scales = [2,2]
        self.decoder_mellstm.upsample_network = UpsampleNetwork(self.upsample_scales)


    def forward(self, mels, mels_translated, spk):  
    
        outputs = {}  
        B = mels.size(0)  

        # Add noise to raw audio
        mels_noisy = mels * (0.02 * torch.randn(mels_translated.shape).cuda()).exp() + 0.003 * torch.randn_like(mels_translated)
       
        # Downsample the mels
        mels_downsampled = self.downsampling_encoder(mels_noisy) 

        # Get approximate phones  
        quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(mels_downsampled.unsqueeze(2))  
        quantized = quantized.squeeze(2)  
 
        # Speaker Embeddings
        embedding = self.spk_embedding(spk)

        # Combine inputs  
        emb = embedding.unsqueeze(1).expand(B, mels_downsampled.shape[1], -1)  
        quantized = torch.cat([quantized, emb], dim=-1) 
    
        # Reconstruction 
        mel_outputs,_ = self.decoder_mellstm(quantized, mels) 
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
 
        # Return 
        return mel_outputs, linear_outputs, vq_penalty.mean(), encoder_penalty.mean(), entropy 


    def forward_getlatents(self, mels):  
        
        B = mels.size(0)  
           
        # Downsample the mels
        mels_downsampled = self.downsampling_encoder(mels) 

        # Get approximate phones  
        latents, entropy = self.quantizer.get_quantizedindices(mels_downsampled.unsqueeze(2))  
        print("Entropy and latents: ", entropy, latents)

    def forward_eval(self, mels, spk):  
        
        outputs = {}  
        B = mels.size(0)  

        # Downsample the mels
        mels_downsampled = self.downsampling_encoder(mels) 

        # Get approximate phones  
        quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(mels_downsampled.unsqueeze(2))  
        quantized = quantized.squeeze(2)  
        print("Entropy is ", entropy)
 
        # Speaker Embeddings
        embedding = self.spk_embedding(spk)

        # Combine inputs  
        emb = embedding.unsqueeze(1).expand(B, mels_downsampled.shape[1], -1)  
        quantized = torch.cat([quantized, emb], dim=-1) 
    
        # Reconstruction 
        mel_outputs = self.decoder_mellstm.forward_generate(quantized)
        print("Shape of mel_outputs: ", mel_outputs.shape)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
 
        # Return 
        return mel_outputs, linear_outputs #, vq_penalty.mean(), encoder_penalty.mean(), entropy 


