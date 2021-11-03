import sys
import time
import multiprocessing
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from math import log10
import torch
import torch.nn.functional as F
from Utils.SSIM import ssim
from Utils.GenomeDISCO import compute_reproducibility
from Utils.io import spreadM, together

from Arg_Parser import *


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


def hicarn_predictor(model, hicarn_loader, ckpt_file, device):
    deepmodel = model.Generator().to(device)
    if not os.path.isfile(ckpt_file):
        ckpt_file = f'save/{ckpt_file}'
    deepmodel.load_state_dict(torch.load(ckpt_file))
    print(f'Loading HiCSR checkpoint file from "{ckpt_file}"')

    result_data = []
    result_inds = []
    test_metrics = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
    ssims = []
    psnrs = []
    mses = []
    repro = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(hicarn_loader, desc='HiCSR Predicting: '):
            lr, hr, inds = batch
            batch_size = lr.size(0)
            test_metrics['nsamples'] += batch_size
            lr = lr.to(device)
            hr = hr.to(device)
            out = predict(deepmodel, lr)

            batch_mse = ((out - hr) ** 2).mean()
            test_metrics['mse'] += batch_mse * batch_size
            batch_ssim = ssim(out, hr)
            test_metrics['ssims'] += batch_ssim * batch_size
            test_metrics['psnr'] = 10 * log10(1 / (test_metrics['mse'] / test_metrics['nsamples']))
            test_metrics['ssim'] = test_metrics['ssims'] / test_metrics['nsamples']
            tqdm(hicarn_loader, desc='HiCNN Predicting: ').set_description(
                desc=f"[Predicting in Test set] PSNR: {test_metrics['psnr']:.4f} dB SSIM: {test_metrics['ssim']:.4f}")

            for i, j in zip(hr, out):
                out1 = torch.squeeze(j, dim=0)
                hr1 = torch.squeeze(i, dim=0)
                out2 = out1.cpu().detach().numpy()
                hr2 = hr1.cpu().detach().numpy()
                genomeDISCO = compute_reproducibility(out2, hr2, transition=True)
                repro.append(genomeDISCO)

            ssims.append(test_metrics['ssim'])
            psnrs.append(test_metrics['psnr'])
            mses.append(batch_mse)

            result_data.append(out.to('cpu').numpy())
            result_inds.append(inds.numpy())
    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0)
    mean_ssim = sum(ssims) / len(ssims)
    mean_psnr = sum(psnrs) / len(psnrs)
    mean_mse = sum(mses) / len(mses)
    mean_repro = sum(repro) / len(repro)

    print("Mean SSIM: ", mean_ssim)
    print("Mean PSNR: ", mean_psnr)
    print("Mean MSE: ", mean_mse)
    print("GenomeDISCO Score: ", mean_repro)
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


    pool = multiprocessing.Pool(processes=pool_num)
    print(f'Start a multiprocess pool with process_num = 3 for saving predicted data')
    for key in compacts.keys():
        pool.apply_async(save_data_n, (key,))
    pool.close()
    pool.join()
    print(f'All data saved. Running cost is {(time.time() - start) / 60:.1f} min.')
