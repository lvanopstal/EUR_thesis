import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import math
import pandas as pd
from scipy.optimize import minimize
import fastai
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.tri import Triangulation
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


class IVSDataset(Dataset):

    def __init__(self, path, transform=None):
        # Load the data
        # TODO --> the arg 'usecols' in self.data is hardcoded, this should ALWAYS refer to implied vol
        self.data = np.loadtxt(path, delimiter=",", dtype=np.float32,
                               skiprows=1, usecols=(6,))
        self.dates = pd.read_csv(path, usecols=['date'])
        self.transform = transform

    def __len__(self):
        # Each date has 40 observations
        return len(self.data) // 40

    def __getitem__(self, idx):
        # Extract the rows corresponding to the idx-th date
        start_idx = idx * 40
        end_idx = start_idx + 40

        # Makes sure to obtain correct data if indexing happens from the back with negative vals
        if end_idx == 0:
            sample = self.data[start_idx:]
        else:
            sample = self.data[start_idx:end_idx]

        # Extract features and target from the sample
        features = sample[:]
        target = sample

        # Convert to tensor
        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        # Obtain the corresponding string date
        date = self.dates.iloc[start_idx][0]

        if self.transform:
            features = self.transform(features)

        return features, target


class Autoencoder(nn.Module):
    def __init__(self, architecture, latent_size):
        self.architecture = architecture
        self.latent_size = latent_size
        # N, 40
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # The length of the architecture
        enc_dec_len = int((len(architecture) - 1) / 2)

        # Separate the architecture of decoder and encoder
        encoder_arch = architecture[:enc_dec_len]
        decoder_arch = architecture[enc_dec_len + 1:]

        # Encoder
        for i in range(enc_dec_len):
            if i == 0:
                self.encoder_layers.append(nn.Linear(input_size, encoder_arch[i]))
                self.encoder_layers.append(nn.Tanh())
                self.encoder_layers.append(nn.BatchNorm1d(encoder_arch[i]))
            else:
                self.encoder_layers.append(nn.Linear(encoder_arch[i - 1], encoder_arch[i]))
                self.encoder_layers.append(nn.Tanh())
                self.encoder_layers.append(nn.BatchNorm1d(encoder_arch[i]))

        # Create the latent layer
        self.encoder_layers.append(nn.Linear(encoder_arch[-1], latent_size))
        self.encoder_layers.append(nn.Tanh())
        self.encoder_layers.append(nn.BatchNorm1d(latent_size))

        # Decoder
        for i in range(enc_dec_len):
            if i == 0:
                self.decoder_layers.append(nn.Linear(latent_size, decoder_arch[i]))
                self.decoder_layers.append(nn.Tanh())
                self.decoder_layers.append(nn.BatchNorm1d(decoder_arch[i]))
            else:
                self.decoder_layers.append(nn.Linear(decoder_arch[i - 1], decoder_arch[i]))
                self.decoder_layers.append(nn.Tanh())
                self.decoder_layers.append(nn.BatchNorm1d(decoder_arch[i]))


        # Build output layer
        self.decoder_layers.append(nn.Linear(decoder_arch[-1], 40))
        self.decoder_layers.append(nn.Sigmoid())

        # Convert to Sequential modules
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        # encoded = model.add_contract_to_z(encoded,)
        print(f'latent vec of current sample: {encoded}')
        decoded = self.decoder(encoded)
        return decoded

    def get_z(self, x):
        z = self.encoder(x)
        return z



def create_architecture(width, latent_size):
    # For now the the network architecture is fixed for a width of 2,3 and 4 hidden layers (excluded the latent z space)
    depth = []
    if width == 3:
        depth = [32, latent_size, 32]
    elif width == 5:
        depth = [32, 16, latent_size, 16, 32]
    elif width == 7:
        depth = [32, 16, 8, latent_size, 8, 16, 32]
    else:
        print('Width should be of size 3, 5 or 7')
        sys.exit()
    return depth



def training(eta, lr, wd, mdl, data, tr_criterion, output_df=None):
    outputs = []
    columns = []
    avg_epoch_loss_list = []
    mdl.train()
    optim = torch.optim.Adam(mdl.parameters(), lr=lr, weight_decay=wd)

    # This creates a seperate list of column names based on the latent dimension
    for l in range(mdl.latent_size):
        columns.append(f'Z_{l+1}')

    # Create the Dataframe that keeps track of all the z-vals
    z_df = pd.DataFrame(columns=columns)

    # Start training
    for epoch in range(eta):
        loss_list = []
        for (img, _) in data:


            # First create an empty tensor per sample that stores the single points of the reconstructed surface
            reconstructed_surf = mdl(img)
            loss = tr_criterion(reconstructed_surf, img)

            # Obtain the z-value for this sample
            z = model.get_z(img).detach().numpy()

            # Convert the NumPy array to a temporary DataFrame
            temp_df = pd.DataFrame(z, columns=columns)

            # Append the temporary DataFrame to the main DataFrame
            z_df = z_df.append(temp_df, ignore_index=True)


            # Print gradients
            print("Gradients:")
            for name, param in mdl.named_parameters():
                if param.grad is not None:
                    print(f"{name}: {param.grad.data.sum()}")
                # print(name, param.data)


            print(f'Loss of current sample: {loss.item():.7f} of epoch: {epoch+1}')
            optim.zero_grad()
            loss.backward()
            optim.step()
            outputs.append((epoch, img, reconstructed_surf))
            loss_list.append(loss.item())
            avg_epoch_loss = sum(loss_list)/ len(loss_list)
        print(f'Epoch: {epoch + 1}, Loss:{avg_epoch_loss:.4f}')
        avg_epoch_loss_list.append(avg_epoch_loss)

    # Write the z values used in training to a column
    z_df.to_csv(f'data/to_use/output/{wise}/{ticker}/TRAINING_z_values{model.latent_size}_{model_version}')

    return avg_epoch_loss_list


def calibrate(model, subset_size, latent_size, num_iterations, dataloader):
    model.eval()
    # Obtain the training data
    cal_outputs = []
    # Define the calibration criterion
    cal_criterion = nn.MSELoss()

    for (img, _) in dataloader:
        z_init = np.zeros(latent_size)

        # Sample x random points
        indices = np.random.choice(40, size=subset_size, replace=False)
        known_points = img.squeeze()[indices]

        def objective(z):
            z = torch.tensor(z, dtype=torch.float32, requires_grad=False).view(1, -1)
            recon_full_surface = model.decoder(z).squeeze()
            recon_points = recon_full_surface[indices]
            loss = cal_criterion(recon_points, known_points)
            return loss.item()

        result = minimize(objective, z_init, method='Nelder-Mead', options={'maxiter': num_iterations})

        z_optimized = torch.tensor(result.x, dtype=torch.float32, requires_grad=False).view(1, -1)
        recon_optimized = model.decoder(z_optimized).squeeze()

        loss_optimized = cal_criterion(recon_optimized[indices], known_points).item()

        print(f'Optimization result: {result.message}')
        print(f'Optimized z values are {z_optimized}')
        print(f'Final Loss: {loss_optimized:.6f}')

        cal_outputs.append((indices, z_optimized, recon_optimized, known_points, loss_optimized))

    return cal_outputs



# Data preprocessing
ticker_list = ['MSFT', 'TSLA', 'XOM', 'SPX']
ticker = ticker_list[3]
wise = 'gridwise'
path = f'data/to_use/{ticker}'
model_version = 'v7'
df = pd.read_csv(path)

# Make train test split
if ticker == 'TSLA':
    train_start = '2010-07-08'
else:
    train_start = '2010-01-04'
validation_start = '2019-01-02'
test_start = '2020-01-02'

# Split the DataFrame based on the found index
val_split_index = df[df['date'] == validation_start].index.min()
test_split_index = df[df['date'] == test_start].index.min()
train_df = df.loc[:val_split_index - 1]
val_df = df.loc[val_split_index: test_split_index -1]
test_df = df.loc[test_split_index:]

train_df.to_csv(f'{path}_train')
val_df.to_csv(f'{path}_val')
test_df.to_csv(f'{path}_test')

# Define the grid points and create a dictionary for later reference
tenor_points = [30, 60, 91, 182, 273, 365, 547, 730]
delta_points = [10, 25, 50, 75, 90]


tmin, tmax = min(tenor_points), max(tenor_points)
for i, val in enumerate(tenor_points):
    tenor_points[i] = (val-tmin) / (tmax-tmin)

dmin, dmax = min(delta_points), max(delta_points)
for i, val in enumerate(delta_points):
    delta_points[i] = (val-dmin) / (dmax-dmin)

grid_points = [tenor_points, delta_points]


# Flatten the grid_points to get a list of all pairs
all_pairs = [(x, y) for x in grid_points[0] for y in grid_points[1]]

# Create a dictionary with keys from 0 to 39 and corresponding grid points
grid_dict = {i: pair for i, pair in enumerate(all_pairs)}

# Create a tensorlist with the option contract pair
tensor_list = [torch.tensor(value) for value in grid_dict.values()]
contract_tensor = torch.stack(tensor_list)

tr = pd.read_csv(f'{path}_train')
val = pd.read_csv(f'{path}_val')
ts = pd.read_csv(f'{path}_test')

tr_path = f'data/to_use/{ticker}_train'
val_path = f'data/to_use/{ticker}_val'
ts_path = f'data/to_use/{ticker}_test'
train_dataset = IVSDataset(tr_path)
val_dataset = IVSDataset(val_path)
test_dataset = IVSDataset(ts_path)
tr_batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=tr_batch_size, shuffle=False, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=tr_batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1,shuffle=False)  # TODO --> the batchsize of the testdataset is fixed to one, possibly change

# Use dataloader
train_dataiter = iter(train_dataloader)
val_dataiter = iter(val_dataloader)
test_dataiter = iter(test_dataloader)

# note that for this use-case the 'images' are the same as the 'labels'
tr_images, tr_labels = next(train_dataiter)
val_images, val_labels = next(val_dataiter)
ts_images, ts_labels = next(test_dataiter)

# Input size is fixed
input_size = 40

### HYPER PARAMETER TRAINING SEARCH ###
to_train_hyper_search = False

if to_train_hyper_search:
    weight_decay_list = [1e-3, 1e-2, 1e-1]
    width = 7
    max_epochs = 5
    lr_steps = [1e-1, 1e-2, 1e-3, 1e-4]
    latent_size_list = [1, 2, 3, 4]
    col_names = [f"eta{eta}" for eta in range(1, max_epochs + 1)]
    row_names = [f"wd{wd}, l{l}, lr{lr}" for wd in weight_decay_list for l in latent_size_list
                 for lr in lr_steps]
    tr_loss_df = pd.DataFrame(index=row_names, columns=col_names)

    for wd in weight_decay_list:
        for lr in lr_steps:
            for l in latent_size_list:
                architecture = create_architecture(width, l)
                model = Autoencoder(architecture, l)
                losses = training(max_epochs, lr, wd=wd, mdl=model, data=val_dataloader,
                                  tr_criterion=nn.MSELoss())

                tr_loss_df.loc[f"wd{wd}, l{l}, lr{lr}"] = losses

    tr_loss_df.to_csv(f'{path}_hyperparameter_losses_{wise}_{model_version}')


### TRAINING ###
to_train = False
if to_train:
    # Create the model
    # These values are based on the above hyper parameter search
    latent_size_list = [1, 2, 3, 4]
    for l in latent_size_list:
        width = 7 #TODO --> KEEP FIXED AFTER HYPER SEARCH
        latent_size = l
        epochs = 4 #TODO --> KEEP FIXED AFTER HYPER SEARCH
        weight_decay = 0.001 #TODO --> KEEP FIXED AFTER HYPER SEARCH
        learning_rate = 0.01 #TODO --> KEEP FIXED AFTER HYPER SEARCH
        architecture = create_architecture(width, latent_size)
        model = Autoencoder(architecture, latent_size)
        training(eta=epochs, lr=learning_rate, wd=weight_decay, mdl=model, data=train_dataloader, tr_criterion=nn.MSELoss())
        torch.save(model, f'data/to_use/models/{wise}/{ticker}/{ticker}_model_latentdim{l}_width7_{model_version}')

### CALIBRATION / FORECASTING ###
to_calibrate = True

if to_calibrate:
    latent_size_list = [1, 2, 3, 4]
    subset_size_list = [5, 10, 15, 20, 25, 30, 35, 40]
    for l in latent_size_list:


        width = 7
        model = torch.load(f'data/to_use/models/{wise}/{ticker}/{ticker}_model_latentdim{l}_width7_{model_version}')

        model.eval()
        for s in subset_size_list:
            subset_size = s
            cal_outputs = calibrate(model, subset_size=subset_size, latent_size=l, num_iterations=100, dataloader=test_dataloader)
            # Make a dataframe containing the calibrated/forecast outputs
            fc_df = pd.DataFrame()
            date_list, indices_list, z_list, fc_list, known_list, loss_list = [], [], [], [], [], []

            # Fill the dataframe by looping through the output, which is in the form of a tuple list
            for ele in cal_outputs:
                indices_list.append(ele[0])
                z_list.append(ele[1].detach().numpy())
                fc_list.append(ele[2].detach().numpy())
                known_list.append(ele[3].detach().numpy())
                loss_list.append(ele[4])
            lossfinal = sum(loss_list)/len(loss_list)
            print(f' final loss is {lossfinal}  for l:{l}, s:{s}')
            # Fill the dataframe
            unique_dates = test_df['date'].drop_duplicates()
            fc_df['date'] = unique_dates
            fc_df['indices'] = indices_list
            fc_df['z'] = z_list
            fc_df['fc_vol'] = fc_list
            fc_df['known_vol_ss'] = known_list
            fc_df['loss_of_ss'] = loss_list

            # Save the dataframe
            fc_df.to_csv(f'data/to_use/output/{wise}/{ticker}/calibration_{ticker}_ss{subset_size}_ld{l}_{model_version}')






