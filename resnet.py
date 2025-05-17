import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# --- Configuration ---
PT_FILES_DIR = "matrix"
OUTPUT_MODEL_DIR = "cvae_pytorch_models_AD_CN_classifier_resnet_v1" # New output dir
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

# Model Hyperparameters
IMG_CHANNELS = 3
IMG_SIZE = 116
LATENT_DIM = 128 * 2  # 256 (User's preference)

# --- ResNet-style Encoder Parameters ---
# Initial conv layer parameters
INITIAL_CONV_FILTERS = 64
INITIAL_KERNEL_SIZE = 7
INITIAL_STRIDE = 2
INITIAL_PADDING = 3 # (kernel_size - 1) // 2 to maintain size with stride 1, or adjust for stride 2

# Max pooling after initial conv
INITIAL_MAX_POOL = True # Whether to use MaxPooling after the first conv block

# ResNet block configurations: list of tuples (num_blocks, filters_list, stride_for_first_conv_block)
# filters_list is [filters1, filters2, filters3] for the bottleneck design
# Stride for the first conv_block in a stage (usually 2 for downsampling, 1 otherwise)
RESNET_STAGES_PARAMS = [
    (2, [64, 64, 256], 1),    # Stage 2 (after initial conv/pool) - no downsampling here
    (2, [128, 128, 512], 2),  # Stage 3 - downsampling
    (2, [256, 256, 1024], 2), # Stage 4 - downsampling
    # (2, [512, 512, 2048], 2) # Stage 5 - optional, might be too deep for 116x116
]
RESNET_KERNEL_SIZE = 3 # Kernel size for the middle conv in bottleneck blocks

ENCODER_DENSE_UNITS = [512] # Dense units after ResNet blocks & avg pooling

# --- Decoder Parameters (will need to be compatible with ResNet encoder output) ---
# The decoder will be a simpler ConvTranspose stack for now, 
# but its starting dimensions depend on the ResNet encoder's output.
# We will calculate this dynamically.
DECODER_CONVTRANSPOSE_LAYERS_PARAMS = [ 
    (128, 4, 2, 1, 0), 
    (64, 4, 2, 1, 0),
    (32, 4, 2, 1, 0)
]
FINAL_DECODER_CONVTRANSPOSE_PARAMS = (IMG_CHANNELS, 4, 2, 1, 0) 
DECODER_DENSE_UNITS = [512] # To match ENCODER_DENSE_UNITS for symmetry before reshape

# Training Hyperparameters
CVAE_EPOCHS = 200 # Adjusted as ResNet might train differently
CVAE_BATCH_SIZE = 32
CVAE_LEARNING_RATE = 1e-4
BETA_START = 0.001
BETA_END = 1.0
BETA_ANNEAL_EPOCHS = 75

# Classifier Hyperparameters
CLASSIFIER_DENSE_UNITS = [128, 64]
CLASSIFIER_DROPOUT_RATE = 0.5
CLASSIFIER_EPOCHS = 150
CLASSIFIER_LR = 5e-4
CLASSIFIER_WEIGHT_DECAY = 5e-4
NUM_BINARY_CLASSES = 2

EARLY_STOPPING_PATIENCE_CVAE = 20 # Adjusted
EARLY_STOPPING_PATIENCE_CLF = 20 # Adjusted

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- EarlyStopping Class (remains the same) ---
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, mode='min'):
        self.patience = patience; self.verbose = verbose; self.counter = 0
        self.best_score = None; self.early_stop = False
        self.val_metric_best = np.Inf if mode == 'min' else -np.Inf
        self.delta = delta; self.path = path; self.trace_func = trace_func
        self.mode = mode

    def __call__(self, current_val_metric, model):
        score = -current_val_metric if self.mode == 'min' else current_val_metric
        if self.best_score is None:
            self.best_score = score; self.save_checkpoint(current_val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score; self.save_checkpoint(current_val_metric, model); self.counter = 0

    def save_checkpoint(self, current_val_metric, model):
        if self.verbose: 
            if self.mode == 'min':
                self.trace_func(f'Validation loss decreased ({self.val_metric_best:.6f} --> {current_val_metric:.6f}). Saving model ...')
            else: # mode == 'max'
                self.trace_func(f'Validation metric improved ({self.val_metric_best:.6f} --> {current_val_metric:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_metric_best = current_val_metric

# --- 1. Data Loading and Preprocessing (ConnectomeDataset class remains the same) ---
class ConnectomeDataset(Dataset):
    def __init__(self, pt_files_dir, selected_metrics, pathology_mapping, 
                 condition_preprocessors_input=None, data_scalers_input=None, 
                 num_subjects_limit=None, fit_scalers=True):
        self.pt_files = sorted(glob.glob(os.path.join(pt_files_dir, "*.pt")))
        if num_subjects_limit: self.pt_files = self.pt_files[:num_subjects_limit]

        self.selected_metrics = selected_metrics
        self.pathology_mapping = pathology_mapping
        self.pathology_categories = ["AD", "CN", "Other"] 
        
        self.matrices, self.subject_ids, self.ages_raw, self.sexes_raw, self.pathologies_mapped_raw = [], [], [], [], []
        
        print(f"Loading data for {len(self.pt_files)} subjects...")
        for file_path in self.pt_files:
            try:
                content = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False) 
                meta = content["meta"]; data_tensor_all_metrics = content["data"] 
                original_group = meta.get("Group"); mapped_group = self.pathology_mapping.get(original_group)
                if mapped_group is None: continue

                metric_indices = [meta["MetricsOrder"].index(m) for m in self.selected_metrics]
                self.matrices.append(data_tensor_all_metrics[metric_indices, :, :])
                self.ages_raw.append(float(meta.get("Age", np.nan)))
                self.sexes_raw.append(meta.get("Sex", "Unknown")) 
                self.pathologies_mapped_raw.append(mapped_group)
                self.subject_ids.append(meta.get("SubjectID"))
            except Exception as e: print(f"Skipping file {file_path} due to error: {e}")
        
        if not self.matrices: raise ValueError("No data loaded.")
        self.matrices = torch.stack(self.matrices).float()
        
        ages_np_temp = np.array(self.ages_raw).astype(float)
        if fit_scalers: 
            self.mean_age = np.nanmean(ages_np_temp) if np.sum(~np.isnan(ages_np_temp)) > 0 else 70.0
        
        mean_age_to_use = self.mean_age if fit_scalers else condition_preprocessors_input['age_mean']
        ages_np = np.nan_to_num(ages_np_temp, nan=mean_age_to_use).reshape(-1,1)
        sexes_np = np.array(self.sexes_raw).reshape(-1, 1)
        pathologies_ohe_np = np.array(self.pathologies_mapped_raw).reshape(-1,1)

        if fit_scalers:
            self.age_scaler = MinMaxScaler(feature_range=(0,1))
            self.sex_encoder = OneHotEncoder(categories=[['Female', 'Male', 'Unknown']], sparse_output=False, handle_unknown='ignore')
            self.pathology_encoder_ohe = OneHotEncoder(categories=[self.pathology_categories], sparse_output=False, handle_unknown='error')
            
            self.ages_scaled = self.age_scaler.fit_transform(ages_np)
            self.sexes_encoded = self.sex_encoder.fit_transform(sexes_np)
            self.pathologies_encoded_ohe = self.pathology_encoder_ohe.fit_transform(pathologies_ohe_np)
            self.condition_preprocessors = {'age_scaler': self.age_scaler, 'sex_encoder': self.sex_encoder, 
                                            'pathology_ohe_encoder': self.pathology_encoder_ohe, 'age_mean': self.mean_age}
        else: 
            self.condition_preprocessors = condition_preprocessors_input
            self.ages_scaled = self.condition_preprocessors['age_scaler'].transform(ages_np)
            self.sexes_encoded = self.condition_preprocessors['sex_encoder'].transform(sexes_np)
            self.pathologies_encoded_ohe = self.condition_preprocessors['pathology_ohe_encoder'].transform(pathologies_ohe_np)

        self.conditions_combined_for_cvae = torch.tensor(np.concatenate(
            [self.ages_scaled, self.sexes_encoded, self.pathologies_encoded_ohe], axis=1), dtype=torch.float32)

        self.data_scalers = [] if data_scalers_input is None else data_scalers_input
        for i in range(self.matrices.shape[1]):
            channel_data = self.matrices[:, i, :, :].reshape(self.matrices.shape[0], -1)
            if fit_scalers:
                if self.selected_metrics[i] == "GrangerCausality_Directed_FDR": channel_data = torch.log1p(channel_data)
                scaler = MinMaxScaler(feature_range=(0,1))
                self.matrices[:, i, :, :] = torch.tensor(scaler.fit_transform(channel_data.numpy()), dtype=torch.float32).reshape(self.matrices.shape[0], IMG_SIZE, IMG_SIZE)
                self.data_scalers.append(scaler)
            else:
                scaler = self.data_scalers[i]
                if self.selected_metrics[i] == "GrangerCausality_Directed_FDR": channel_data = torch.log1p(channel_data)
                self.matrices[:, i, :, :] = torch.tensor(scaler.transform(channel_data.numpy()), dtype=torch.float32).reshape(self.matrices.shape[0], IMG_SIZE, IMG_SIZE)
        
        self.pathology_labels_str = self.pathologies_mapped_raw

    def __len__(self): return len(self.matrices)
    def __getitem__(self, idx):
        pathology_label_map_3_class = {label: i for i, label in enumerate(self.pathology_categories)}
        numerical_label_3_class = pathology_label_map_3_class[self.pathology_labels_str[idx]]
        binary_label = 1 if self.pathology_labels_str[idx] == "AD" else (0 if self.pathology_labels_str[idx] == "CN" else -1)
        return self.matrices[idx], self.conditions_combined_for_cvae[idx], torch.tensor(numerical_label_3_class, dtype=torch.long), torch.tensor(binary_label, dtype=torch.long), self.pathology_labels_str[idx]
    def get_preprocessors(self): return self.condition_preprocessors, self.data_scalers
    def get_condition_dim(self): return self.conditions_combined_for_cvae.shape[1]


# --- 2. ResNet-style Blocks (PyTorch) ---
class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2)
        
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3)

    def forward(self, x):
        shortcut = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out += shortcut # Add shortcut
        out = self.relu(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride=2):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2)

        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3)

        self.shortcut_conv = nn.Conv2d(in_channels, filters3, kernel_size=1, stride=stride, padding=0, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(filters3)

    def forward(self, x):
        shortcut = self.shortcut_conv(x)
        shortcut = self.shortcut_bn(shortcut)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += shortcut
        out = self.relu(out)
        return out

# --- 3. CVAE Model Definition with ResNet-style Encoder ---
class CVAE_ResNetEncoder(nn.Module):
    def __init__(self, img_channels, img_size, condition_dim, latent_dim,
                 initial_conv_filters, initial_kernel_size, initial_stride, initial_padding, use_max_pool,
                 resnet_stages_params, resnet_kernel_size,
                 encoder_dense_units,
                 decoder_dense_units, decoder_convtranspose_params, final_decoder_convtranspose_params):
        super(CVAE_ResNetEncoder, self).__init__()
        self.latent_dim = latent_dim

        # --- ResNet-style Encoder ---
        encoder_modules = []
        current_channels = img_channels
        current_size = img_size

        # Initial Convolution
        encoder_modules.append(nn.Conv2d(current_channels, initial_conv_filters, initial_kernel_size, initial_stride, initial_padding, bias=False))
        encoder_modules.append(nn.BatchNorm2d(initial_conv_filters))
        encoder_modules.append(nn.ReLU(inplace=True))
        current_channels = initial_conv_filters
        current_size = (current_size + 2 * initial_padding - initial_kernel_size) // initial_stride + 1
        if use_max_pool:
            encoder_modules.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            current_size = (current_size + 2 * 1 - 3) // 2 + 1
        
        print("ResNet Encoder Layer Shapes:")
        print(f"  After Initial Conv/Pool: ({current_channels}, {current_size}, {current_size})")

        # ResNet Stages
        for i, (num_blocks, filters_list, stride_for_first) in enumerate(resnet_stages_params):
            encoder_modules.append(ConvBlock(current_channels, filters_list, resnet_kernel_size, stride=stride_for_first))
            current_channels = filters_list[2] # Output channels of the bottleneck
            if stride_for_first > 1:
                 current_size = (current_size + 2 * 0 - 1) // stride_for_first + 1 # Approx for stride in ConvBlock shortcut
            for _ in range(num_blocks - 1):
                encoder_modules.append(IdentityBlock(current_channels, filters_list, resnet_kernel_size))
            print(f"  After Stage {i+2}: ({current_channels}, {current_size}, {current_size})")


        encoder_modules.append(nn.AdaptiveAvgPool2d((1,1))) # Global Average Pooling
        self.encoder_conv = nn.Sequential(*encoder_modules)
        self.flattened_size = current_channels # After AdaptiveAvgPool2d, spatial dims are 1x1

        encoder_fc_layers_list = []
        fc_input_size = self.flattened_size + condition_dim
        for units in encoder_dense_units:
            encoder_fc_layers_list.extend([nn.Linear(fc_input_size, units), nn.ReLU(inplace=True)])
            fc_input_size = units
        self.encoder_fc = nn.Sequential(*encoder_fc_layers_list)
        self.fc_z_mean = nn.Linear(fc_input_size, latent_dim)
        self.fc_z_log_var = nn.Linear(fc_input_size, latent_dim)

        # --- Decoder (Simplified ConvTranspose stack, needs careful dimension matching) ---
        # This decoder might need significant adjustments if the ResNet encoder output shape is very different
        # For simplicity, we'll try to make the ResNet output a feature map that can be fed into a similar decoder structure
        # The `decoder_start_conv_size` will be small after ResNet (e.g., 1 if AdaptiveAvgPool2d, or ~7-15 if not)
        # If using AdaptiveAvgPool2d -> (channels, 1, 1), then the first FC layer in decoder needs to project to a suitable CxHxW
        
        # Let's assume the ResNet encoder's output (after conv part, before FC for z) is (Batch, Channels_final_resnet, H_final_resnet, W_final_resnet)
        # For simplicity, we'll use a dense layer to project from latent_dim + condition_dim to a shape
        # that can be reshaped and then upsampled by ConvTranspose2d layers.
        # This part is highly dependent on the output of the ResNet encoder.
        # The current DECODER_CONVTRANSPOSE_PARAMS are for an input of ~15x15.
        # If ResNet encoder outputs, say, 7x7, these need to change.
        # Let's calculate the required starting size for the decoder based on its transpose conv layers.
        # We want to end up at IMG_SIZE (116).
        # Working backwards from 116:
        # Layer 3 (FINAL_DECODER_CONVTRANSPOSE_PARAMS): (IMG_CHANNELS, 4, 2, 1, 0) -> Input must be 58x58
        # Layer 2 (DECODER_CONVTRANSPOSE_LAYERS_PARAMS[1]): (32, 4, 2, 1, 0) -> Input must be 29x29
        # Layer 1 (DECODER_CONVTRANSPOSE_LAYERS_PARAMS[0]): (64, 3, 2, 1, 0) -> Input must be 15x15 (approx, (29-1+2*1-3)/2+1 = 14, so 15 with output_padding=0 is tricky, needs kernel 3, op 0 to get 29 from 15, or kernel 4, op 1 for 14->29)
        # Let's adjust decoder params assuming a start of ~7x7 or ~8x8 from a deeper ResNet encoder if we used more stages.
        # For now, using the same decoder structure as before, assuming encoder output is ~15x15.
        # The `current_size` from encoder needs to be passed to decoder.
        
        self.decoder_start_conv_channels = RESNET_STAGES_PARAMS[-1][2] # Channels of last ResNet stage output
        # This current_size is tricky with ResNet. If AdaptiveAvgPool is used, it's 1.
        # If not, it's the spatial dim after the last ResNet stage.
        # For now, let's assume the decoder_fc projects to a fixed starting size for ConvTranspose.
        # For example, let's target a 7x7 or 8x8 start for the ConvTranspose part.
        # ------------- inside CVAE_ResNetEncoder.__init__() -----------------
        # right after you set
        self.decoder_fc_target_h_w = 8          # start from 8×8
        self.decoder_start_conv_channels = RESNET_STAGES_PARAMS[-1][2]  # 1024

        # 1) dense projection  (creates self.decoder_fc)  ⬇️⬇️⬇️
        decoder_fc_layers = []
        in_feat  = latent_dim + condition_dim
        out_feat = (self.decoder_start_conv_channels *
                    self.decoder_fc_target_h_w * self.decoder_fc_target_h_w)

        for units in decoder_dense_units:               # e.g. [512]
            decoder_fc_layers += [
                nn.Linear(in_feat, units),
                nn.ReLU(inplace=True)
            ]
            in_feat = units                              # chain them

        decoder_fc_layers += [nn.Linear(in_feat, out_feat), nn.ReLU(inplace=True)]
        self.decoder_fc = nn.Sequential(*decoder_fc_layers)
        # ------------- dense part done --------------------------------------

        # 2) now keep the ConvTranspose chain you already inserted
        decoder_conv_t_layers_list = []
        C = self.decoder_start_conv_channels  # 1024
        # 8 → 15
        decoder_conv_t_layers_list += [
            nn.ConvTranspose2d(C, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        ]
        C = 512
        # 15 → 29
        decoder_conv_t_layers_list += [
            nn.ConvTranspose2d(C, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        ]
        C = 256
        # 29 → 58
        decoder_conv_t_layers_list += [
            nn.ConvTranspose2d(C, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        ]
        C = 128
        # 58 → 116
        decoder_conv_t_layers_list += [
            nn.ConvTranspose2d(C, IMG_CHANNELS, 4, 2, 1, bias=False),
            nn.Sigmoid()
        ]
        self.decoder_conv_transpose = nn.Sequential(*decoder_conv_t_layers_list)
        # --------------------------------------------------------------------




    def encode(self, x_img, x_cond):
        x = self.encoder_conv(x_img) # Output of ResNet blocks + AdaptiveAvgPool is (B, C, 1, 1)
        x = x.view(x.size(0), -1)    # Flatten to (B, C)
        x_combined = torch.cat([x, x_cond], dim=1)
        x_fc = self.encoder_fc(x_combined)
        return self.fc_z_mean(x_fc), self.fc_z_log_var(x_fc)

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var); eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z, x_cond):
        z_combined = torch.cat([z, x_cond], dim=1); x_fc = self.decoder_fc(z_combined)
        # Reshape to start the ConvTranspose sequence
        x = x_fc.view(x_fc.size(0), self.decoder_start_conv_channels, self.decoder_fc_target_h_w, self.decoder_fc_target_h_w)
        return self.decoder_conv_transpose(x)

    def forward(self, x_img, x_cond):
        z_mean, z_log_var = self.encode(x_img, x_cond)
        z = self.reparameterize(z_mean, z_log_var)
        return self.decode(z, x_cond), z_mean, z_log_var

# (Loss function, CVAE training/validation loops, Classifier, Classifier training/evaluation loops remain the same)
# ... (Copy from your previous script: cvae_loss_function, train_cvae_epoch, validate_cvae_epoch) ...
# ... (LatentSpaceBinaryClassifier, train_binary_classifier_epoch_v2, evaluate_binary_classifier) ...
def cvae_loss_function(x_reconstructed, x_original, z_mean, z_log_var, current_beta_kl):
    recon_loss = nn.MSELoss(reduction='sum')(x_reconstructed, x_original) / x_original.size(0) 
    kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) / x_original.size(0)
    return recon_loss + current_beta_kl * kl_div, recon_loss, kl_div

def train_cvae_epoch(model, dataloader, optimizer, current_beta_kl, device):
    model.train(); train_loss_sum = 0; recon_loss_sum = 0; kl_loss_sum = 0
    for matrices, conditions, _, _, _ in dataloader:
        matrices, conditions = matrices.to(device), conditions.to(device)
        optimizer.zero_grad()
        reconstructed, z_mean, z_log_var = model(matrices, conditions)
        loss, recon_l, kl_l = cvae_loss_function(reconstructed, matrices, z_mean, z_log_var, current_beta_kl)
        loss.backward(); optimizer.step()
        train_loss_sum += loss.item(); recon_loss_sum += recon_l.item(); kl_loss_sum += kl_l.item()
    return train_loss_sum/len(dataloader), recon_loss_sum/len(dataloader), kl_loss_sum/len(dataloader)

def validate_cvae_epoch(model, dataloader, current_beta_kl, device, scheduler=None):
    model.eval(); val_loss_sum = 0; recon_loss_sum = 0; kl_loss_sum = 0
    with torch.no_grad():
        for matrices, conditions, _, _, _ in dataloader:
            matrices, conditions = matrices.to(device), conditions.to(device)
            reconstructed, z_mean, z_log_var = model(matrices, conditions)
            loss, recon_l, kl_l = cvae_loss_function(reconstructed, matrices, z_mean, z_log_var, current_beta_kl)
            val_loss_sum += loss.item(); recon_loss_sum += recon_l.item(); kl_loss_sum += kl_l.item()
    avg_val_loss = val_loss_sum/len(dataloader) if len(dataloader) > 0 else 0
    if scheduler: scheduler.step(avg_val_loss)
    return avg_val_loss, recon_loss_sum/len(dataloader) if len(dataloader) > 0 else 0, kl_loss_sum/len(dataloader) if len(dataloader) > 0 else 0

class LatentSpaceBinaryClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes, hidden_units_list, dropout_rate=0.5):
        super(LatentSpaceBinaryClassifier, self).__init__()
        all_units = [latent_dim] + hidden_units_list 
        layers_list = []
        for i in range(len(all_units)): 
            if i < len(all_units) -1 : 
                layers_list.extend([
                    nn.Linear(all_units[i], all_units[i+1]), 
                    nn.BatchNorm1d(all_units[i+1]), 
                    nn.ReLU(inplace=True), 
                    nn.Dropout(dropout_rate)
                ])
            else: 
                 layers_list.append(nn.Linear(all_units[i], num_classes))
        self.network = nn.Sequential(*layers_list)
    def forward(self, latent_vector): return self.network(latent_vector)

def train_binary_classifier_epoch_v2(classifier, dataloader, optimizer, criterion, device):
    classifier.train(); total_loss = 0; correct_predictions = 0; total_samples = 0
    for latent_z, binary_labels in dataloader:
        latent_z, binary_labels = latent_z.to(device), binary_labels.to(device)
        optimizer.zero_grad(); outputs = classifier(latent_z); loss = criterion(outputs, binary_labels)
        loss.backward(); optimizer.step()
        total_loss += loss.item(); _, predicted = torch.max(outputs.data, 1)
        total_samples += binary_labels.size(0); correct_predictions += (predicted == binary_labels).sum().item()
    return total_loss/len(dataloader) if len(dataloader) > 0 else 0, correct_predictions/total_samples if total_samples > 0 else 0

def evaluate_binary_classifier(classifier, dataloader, criterion, device, scheduler=None, is_val_for_scheduler=False):
    classifier.eval(); total_loss = 0; correct_predictions = 0; total_samples = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for latent_z, binary_labels in dataloader:
            latent_z, binary_labels = latent_z.to(device), binary_labels.to(device)
            outputs = classifier(latent_z); loss = criterion(outputs, binary_labels)
            total_loss += loss.item(); _, predicted = torch.max(outputs.data, 1)
            total_samples += binary_labels.size(0); correct_predictions += (predicted == binary_labels).sum().item()
            all_preds.extend(predicted.cpu().numpy()); all_labels.extend(binary_labels.cpu().numpy())
    avg_loss = total_loss/len(dataloader) if len(dataloader) > 0 else 0
    avg_acc = correct_predictions/total_samples if total_samples > 0 else 0
    if scheduler and is_val_for_scheduler: scheduler.step(avg_loss)
    return avg_loss, avg_acc, all_labels, all_preds

# --- Main Execution ---
if __name__ == "__main__":
    selected_metrics = ["Correlation_FisherZ", "NMI", "GrangerCausality_Directed_FDR"]
    pathology_mapping = {"AD": "AD", "CN": "CN", "MCI": "Other", "LMCI": "Other", "EMCI": "Other"}

    full_dataset = ConnectomeDataset(PT_FILES_DIR, selected_metrics, pathology_mapping, fit_scalers=True)
    if len(full_dataset) == 0: print("No data loaded. Exiting."); exit()
    
    condition_preprocessors, data_scalers = full_dataset.get_preprocessors()
    CONDITION_DIM = full_dataset.get_condition_dim()

    total_count = len(full_dataset)
    test_ratio = 0.20
    val_ratio = 0.20 
    
    num_test = int(total_count * test_ratio)
    num_remaining = total_count - num_test
    num_val = int(num_remaining * val_ratio)
    num_train = num_remaining - num_test

    # Create indices for splitting to ensure consistent subjects across CVAE and Classifier datasets
    indices = list(range(total_count))
    np.random.seed(42) # for reproducibility
    np.random.shuffle(indices)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train : num_train + num_val]
    test_indices = indices[num_train + num_val :]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset_final_holdout = torch.utils.data.Subset(full_dataset, test_indices)
    
    train_loader_cvae = DataLoader(train_dataset, batch_size=CVAE_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader_cvae = DataLoader(val_dataset, batch_size=CVAE_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    print(f"CVAE Train set: {len(train_dataset)}, CVAE Validation set: {len(val_dataset)}, Test set: {len(test_dataset_final_holdout)}")

    cvae_model = CVAE_ResNetEncoder( # Using the new ResNet Encoder CVAE
        img_channels=IMG_CHANNELS, img_size=IMG_SIZE, condition_dim=CONDITION_DIM, latent_dim=LATENT_DIM,
        initial_conv_filters=INITIAL_CONV_FILTERS, initial_kernel_size=INITIAL_KERNEL_SIZE,
        initial_stride=INITIAL_STRIDE, initial_padding=INITIAL_PADDING, use_max_pool=INITIAL_MAX_POOL,
        resnet_stages_params=RESNET_STAGES_PARAMS, resnet_kernel_size=RESNET_KERNEL_SIZE,
        encoder_dense_units=ENCODER_DENSE_UNITS,
        decoder_dense_units=DECODER_DENSE_UNITS, 
        # The decoder params below might need significant rework to match the ResNet encoder's output shape
        # For now, using the simplified ones, but this is a key area for adjustment.
        decoder_convtranspose_params=DECODER_CONVTRANSPOSE_LAYERS_PARAMS, 
        final_decoder_convtranspose_params=FINAL_DECODER_CONVTRANSPOSE_PARAMS
    ).to(DEVICE)

    with torch.no_grad():
        dummy_x = torch.zeros(1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE).to(DEVICE)
        dummy_c = torch.zeros(1, CONDITION_DIM).to(DEVICE)
        recon, _, _ = cvae_model(dummy_x, dummy_c)
    print("reconstruction shape:", recon.shape)   # → (1, 3, 116, 116)

    
    optimizer_cvae = optim.Adam(cvae_model.parameters(), lr=CVAE_LEARNING_RATE)
    scheduler_cvae = ReduceLROnPlateau(optimizer_cvae, mode='min', factor=0.2, patience=10, verbose=True) 
    cvae_early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE_CVAE, verbose=True, 
                                       path=os.path.join(OUTPUT_MODEL_DIR, "cvae_resnet_best_model.pth"), mode='min')

    print("\n--- Training CVAE with ResNet Encoder (on AD, CN, Other) ---")
    # (CVAE training loop as before)
    cvae_train_losses_log, cvae_val_losses_log = [], []
    for epoch in range(1, CVAE_EPOCHS + 1):
        current_beta_kl = min(BETA_END, BETA_START + (BETA_END - BETA_START) * (epoch / BETA_ANNEAL_EPOCHS))
        train_loss, train_recon, train_kl = train_cvae_epoch(cvae_model, train_loader_cvae, optimizer_cvae, current_beta_kl, DEVICE)
        val_loss, val_recon, val_kl = validate_cvae_epoch(cvae_model, val_loader_cvae, current_beta_kl, DEVICE, scheduler=scheduler_cvae)
        
        cvae_train_losses_log.append(train_loss); cvae_val_losses_log.append(val_loss)
        if epoch % 10 == 0 or epoch == CVAE_EPOCHS or epoch == 1:
            print(f"CVAE Epoch {epoch}/{CVAE_EPOCHS}: Beta_KL: {current_beta_kl:.4f} LR: {optimizer_cvae.param_groups[0]['lr']:.1e}\n"
                  f"  Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})\n"
                  f"  Val Loss  : {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
        cvae_early_stopper(val_loss, cvae_model)
        if cvae_early_stopper.early_stop: print("CVAE Early stopping triggered."); break
    
    print("Loading best CVAE model weights based on validation loss.")
    cvae_model.load_state_dict(torch.load(os.path.join(OUTPUT_MODEL_DIR, "cvae_resnet_best_model.pth")))
    plt.figure(figsize=(10,5)); plt.plot(cvae_train_losses_log, label="Train CVAE Loss"); plt.plot(cvae_val_losses_log, label="Val CVAE Loss")
    if cvae_val_losses_log: plt.axvline(np.argmin(cvae_val_losses_log), color='r', linestyle='--', label=f'Best Val Epoch: {np.argmin(cvae_val_losses_log)+1}')
    plt.title("CVAE ResNet Training & Validation Loss"); plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_MODEL_DIR, "cvae_resnet_loss_plot.png")); plt.close()


    print("\n--- Preparing Data for AD vs CN Classifier ---")
    cvae_model.eval() 
    def extract_latent_and_filter(dataset_subset):
        latent_vectors_list, binary_labels_list = [], []
        for i in range(len(dataset_subset)): 
            matrix, condition, _, binary_label, pathology_str = dataset_subset[i] 
            if pathology_str in ["AD", "CN"]:
                matrix, condition = matrix.unsqueeze(0).to(DEVICE), condition.unsqueeze(0).to(DEVICE)
                with torch.no_grad(): z_mean, _ = cvae_model.encode(matrix, condition)
                latent_vectors_list.append(z_mean.squeeze(0).cpu())
                binary_labels_list.append(binary_label.cpu()) 
        if not latent_vectors_list: return None, None
        return torch.stack(latent_vectors_list), torch.tensor(binary_labels_list, dtype=torch.long)

    train_latent_clf, train_labels_clf = extract_latent_and_filter(train_dataset)
    val_latent_clf, val_labels_clf = extract_latent_and_filter(val_dataset) 
    test_latent_clf, test_labels_clf = extract_latent_and_filter(test_dataset_final_holdout)

    if train_latent_clf is None or val_latent_clf is None or test_latent_clf is None:
        print("Insufficient AD/CN data in one or more splits for classifier. Exiting."); exit()

    print(f"AD/CN Classifier: Train size={len(train_latent_clf)}, Val size={len(val_latent_clf)}, Test size={len(test_latent_clf)}")
    print(f"Train AD/CN dist: CN={(train_labels_clf==0).sum().item()}, AD={(train_labels_clf==1).sum().item()}")
    print(f"Val AD/CN dist: CN={(val_labels_clf==0).sum().item()}, AD={(val_labels_clf==1).sum().item()}")
    print(f"Test AD/CN dist: CN={(test_labels_clf==0).sum().item()}, AD={(test_labels_clf==1).sum().item()}")

    train_dataset_clf_final = TensorDataset(train_latent_clf, train_labels_clf)
    val_dataset_clf_final = TensorDataset(val_latent_clf, val_labels_clf)
    test_dataset_clf_final_eval = TensorDataset(test_latent_clf, test_labels_clf)

    train_loader_clf = DataLoader(train_dataset_clf_final, batch_size=CVAE_BATCH_SIZE, shuffle=True)
    val_loader_clf = DataLoader(val_dataset_clf_final, batch_size=CVAE_BATCH_SIZE, shuffle=False)
    test_loader_clf_final = DataLoader(test_dataset_clf_final_eval, batch_size=CVAE_BATCH_SIZE, shuffle=False) 
    
    binary_classifier = LatentSpaceBinaryClassifier(
        latent_dim=LATENT_DIM, num_classes=NUM_BINARY_CLASSES, hidden_units_list=CLASSIFIER_DENSE_UNITS, dropout_rate=CLASSIFIER_DROPOUT_RATE
    ).to(DEVICE)
    optimizer_binary_clf = optim.Adam(binary_classifier.parameters(), lr=CLASSIFIER_LR, weight_decay=CLASSIFIER_WEIGHT_DECAY)
    scheduler_clf = ReduceLROnPlateau(optimizer_binary_clf, mode='max', factor=0.2, patience=10, verbose=True) # Monitor accuracy
    criterion_binary_clf = nn.CrossEntropyLoss()
    classifier_early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE_CLF, verbose=True, 
                                             path=os.path.join(OUTPUT_MODEL_DIR, "binary_classifier_AD_CN_resnet_best.pth"), 
                                             mode='max') # Monitor val_acc

    print("\n--- Training AD vs CN Classifier on CVAE Latent Space ---")
    clf_train_losses_log, clf_train_accs_log, clf_val_losses_log, clf_val_accs_log = [], [], [], []
    for epoch in range(1, CLASSIFIER_EPOCHS + 1):
        train_loss_clf, train_acc_clf = train_binary_classifier_epoch_v2(binary_classifier, train_loader_clf, optimizer_binary_clf, criterion_binary_clf, DEVICE)
        val_loss_clf, val_acc_clf, _, _ = evaluate_binary_classifier(binary_classifier, val_loader_clf, criterion_binary_clf, DEVICE, scheduler=scheduler_clf, is_val_for_scheduler=True)
        
        clf_train_losses_log.append(train_loss_clf); clf_train_accs_log.append(train_acc_clf)
        clf_val_losses_log.append(val_loss_clf); clf_val_accs_log.append(val_acc_clf)
        if epoch % 10 == 0 or epoch == CLASSIFIER_EPOCHS or epoch == 1:
            print(f"Binary Classifier Epoch {epoch}/{CLASSIFIER_EPOCHS}: LR: {optimizer_binary_clf.param_groups[0]['lr']:.1e}\n"
                  f"  Train Loss: {train_loss_clf:.4f}, Train Acc: {train_acc_clf:.4f}\n"
                  f"  Val Loss  : {val_loss_clf:.4f}, Val Acc  : {val_acc_clf:.4f}")
        
        classifier_early_stopper(val_acc_clf, binary_classifier) # Early stopping on validation accuracy
        if classifier_early_stopper.early_stop: print("Classifier Early stopping triggered."); break
    
    print("Loading best binary classifier model weights based on validation metric.")
    binary_classifier.load_state_dict(torch.load(os.path.join(OUTPUT_MODEL_DIR, "binary_classifier_AD_CN_resnet_best.pth")))
    
    plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1); plt.plot(clf_train_accs_log, label='Train Acc (AD/CN)'); plt.plot(clf_val_accs_log, label='Val Acc (AD/CN)')
    if clf_val_accs_log: plt.axvline(np.argmax(clf_val_accs_log), color='r', linestyle='--', label=f'Best Val Epoch (Acc): {np.argmax(clf_val_accs_log)+1}')
    plt.title('AD/CN Classifier Accuracy'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2); plt.plot(clf_train_losses_log, label='Train Loss (AD/CN)'); plt.plot(clf_val_losses_log, label='Val Loss (AD/CN)')
    if clf_val_losses_log: plt.axvline(np.argmin(clf_val_losses_log), color='r', linestyle='--', label=f'Best Val Epoch (Loss): {np.argmin(clf_val_losses_log)+1}')
    plt.title('AD/CN Classifier Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_MODEL_DIR, "binary_classifier_AD_CN_resnet_training_plot.png")); plt.close()

    print("\n--- Final Evaluation on Test Set (AD vs CN Classifier) ---")
    test_loss_final, test_acc_final, true_test_labels, pred_test_labels = evaluate_binary_classifier(
        binary_classifier, test_loader_clf_final, criterion_binary_clf, DEVICE) 
    
    # Calculate AUC for binary classification
    # Need probability scores for the positive class (AD=1)
    binary_classifier.eval()
    all_probs_ad = []
    with torch.no_grad():
        for latent_z, _ in test_loader_clf_final: # Only need latent_z for prediction
            latent_z = latent_z.to(DEVICE)
            outputs = binary_classifier(latent_z)
            probs = torch.softmax(outputs, dim=1) # Get probabilities
            all_probs_ad.extend(probs[:, 1].cpu().numpy()) # Prob for class 1 (AD)
            
    auc_score = roc_auc_score(true_test_labels, all_probs_ad) # Use probabilities for AUC

    print(f"Final Test Set: Loss = {test_loss_final:.4f}, Accuracy = {test_acc_final:.4f}, AUC = {auc_score:.4f}")
    print("\nFinal Test Set Classification Report (AD vs CN):")
    print(classification_report(true_test_labels, pred_test_labels, target_names=["CN (0)", "AD (1)"], zero_division=0))
    print("\nFinal Test Set Confusion Matrix (AD vs CN):")
    print(confusion_matrix(true_test_labels, pred_test_labels))

    print("--- Script Finished ---")
