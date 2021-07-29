import sys
import time
import multiprocessing
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import Models.HiCSR as hicsr
import Models.HiCNN as hicnn
import Models.HiCPlus as hicplus
import torch
import torch.nn.functional as F
from Utils.io import spreadM, together

from Data.Arg_Parser import *


# Adjust 40x40 data for HiCSR/HiCNN/HiCPlus 28x28 output
def predict(model, data):
    padded_data = F.pad(data, (6, 6, 6, 6), mode='constant')
    predicted_mat = torch.zeros((1, 1, padded_data.shape[2], padded_data.shape[3]))
    predicted_mat = model(padded_data).to(device)
    return predicted_mat


def dataloader(data, batch_size=64):
    inputs = torch.tensor(data['data'], dtype=torch.float)
    target = torch.tensor(data['target'], dtype=torch.float)
    inds = torch.tensor(data['inds'], dtype=torch.long)
    dataset = TensorDataset(inputs, target, inds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def data_info(data):
    indices = data['inds']
    compacts = data['compacts'][()]
    sizes = data['sizes'][()]
    return indices, compacts, sizes


get_digit = lambda x: int(''.join(list(filter(str.isdigit, x))))


def filename_parser(filename):
    info_str = filename.split('.')[0].split('_')[2:-1]
    chunk = get_digit(info_str[0])
    stride = get_digit(info_str[1])
    bound = get_digit(info_str[2])
    scale = 1 if info_str[3] == 'nonpool' else get_digit(info_str[3])
    return chunk, stride, bound, scale


def hicarn_predictor(model, deephic_loader, ckpt_file, device):
    model = model.Generator().to(device)
    if not os.path.isfile(ckpt_file):
        ckpt_file = f'save/{ckpt_file}'
    model.load_state_dict(torch.load(ckpt_file))
    print(f'Loading HiCSR checkpoint file from "{ckpt_file}"')

    result_data = []
    result_inds = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(deephic_loader, desc='HiCSR Predicting: '):
            lr, hr, inds = batch
            lr = lr.to(device)
            out = predict(model, lr)

            result_data.append(out.to('cpu').numpy())
            result_inds.append(inds.numpy())
    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0)
    deep_hics = together(result_data, result_inds, tag='Reconstructing: ')
    return deep_hics


def save_data(hicarn_hic, compact, size, file):
    hicarn = spreadM(hicarn_hic, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, hicarn=hicarn, compact=compact)
    print('Saving file:', file)


if __name__ == '__main__':
    args = data_predict_parser().parse_args(sys.argv[1:])
    cell_line = args.cell_line
    low_res = args.low_res
    ckpt_file = args.checkpoint
    res_num = args.resblock
    cuda = args.cuda
    model = args.model
    print('WARNING: Prediction process requires a large memory. Ensure that your machine has ~150G of memory.')
    if multiprocessing.cpu_count() > 23:
        pool_num = 23
    else:
        exit()

    in_dir = os.path.join(root_dir, 'data')
    out_dir = os.path.join(root_dir, 'predict', cell_line)
    mkdir(out_dir)

    files = [f for f in os.listdir(in_dir) if f.find(low_res) >= 0]
    hicarn_file = os.path.join(root_dir, 'data/', args.file)

    chunk, stride, bound, scale = filename_parser(hicarn_file)

    device = torch.device(
        f'cuda:{cuda}' if (torch.cuda.is_available() and cuda > -1 and cuda < torch.cuda.device_count()) else 'cpu')
    print(f'Using device: {device}')

    start = time.time()
    print(f'Loading data: {hicarn_file}')
    hicarn_data = np.load(os.path.join(in_dir, hicarn_file), allow_pickle=True)
    hicarn_loader = dataloader(hicarn_data)

    indices, compacts, sizes = data_info(hicarn_data)
    hicarn_hics = hicarn_predictor(model, hicarn_loader, ckpt_file, device)


    def save_data_n(key):
        file = os.path.join(out_dir, f'predict_chr{key}_{low_res}.npz')
        save_data(hicarn_hics[key], compacts[key], sizes[key], file)


    pool = multiprocessing.Pool(processes=3)
    print(f'Start a multiprocess pool with process_num = 3 for saving predicted data')
    for key in compacts.keys():
        pool.apply_async(save_data_n, (key,))
    pool.close()
    pool.join()
    print(f'All data saved. Running cost is {(time.time() - start) / 60:.1f} min.')
