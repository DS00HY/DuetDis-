import torch
import sys
from copy import Error, deepcopy
from torch import nn
from DuetDis.Duet import seresnet101
import torch.nn as nn
import random, os
import string, re
import numpy as np
from math import sqrt
import pandas as pd
import torch.nn.functional as F
import argparse
from torch import nn
from multiprocessing import Pool
from functools import  partial


def load_features(feat_path, prot, model_id):
    if model_id in "12":
        feat = np.load(feat_path+"/"+prot+"_features_m1.npz")
    else:
        feat = np.load(feat_path+"/"+prot+"_features_m3.npz")
    return  feat['features']

def dist2con(pred, out_file):
    w = np.sum(pred[:,:,1:13], axis=-1)
    L = w.shape[0]
    idx = np.array([[i+1,j+1,0,8,w[i,j]] for i in range(L) for j in range(i+1,L)])
    out = idx[np.flip(np.argsort(idx[:,4]))]
    data = [out[:,0].astype(int), out[:,1].astype(int), out[:,4].astype(float)]
    df = pd.DataFrame(data)
    df = df.transpose()
    df[0] = df[0].astype(int)
    df[1] = df[1].astype(int)
    df.columns = ["i", "j", "p"]
    df.to_csv(out_file, sep=' ', index=False, header=False)


def predict(args, model_id):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt= args.model
    ckpt = ckpt + "/Duet_N1_M" + model_id +".pth"
    chainlist = args.target
    featpath = args.feat
    outpath1 = args.output #args.cmap
    outpath2 = args.output 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(chainlist,"r") as chainfile:
        protlist = chainfile.readlines()
    protlist = [prot.strip() for prot in protlist]
    soft = nn.Softmax(dim=1)
    if model_id in "12":
        model = seresnet101(in_planes=526)
    else:
        model = seresnet101(in_planes=151)
    model.eval()
    if device == "gpu":
        checkpoint=torch.load(ckpt)
    else:
        checkpoint=torch.load(ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    #checkpoint = torch.load(ckpt)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #model.load_state_dict(torch.load(ckpt))
    model = model.to(device)

    with torch.no_grad():
        for prot in protlist:
            try:
                outfile1 = outpath1 + "/N1_" + prot + model_id + ".npz"
                outfile2 = outpath2 + "/N1_" + prot + model_id + ".map_std"
                #msa = parse_a3m(a3mfile)
                #msa = torch.from_numpy(msa).to(torch.int64)
                #msa = torch.squeeze(msa, dim=0)
                #featfile = featpath+"/"+prot+"/"+prot+"_features.npz"
                featfile = featpath+"/"+prot+"/"
                if os.path.exists(outfile2):
                    print(prot+" contact map exist!")
                elif os.path.exists(featfile):
                    f2d = torch.tensor(load_features(featfile, prot, model_id))
                    f2d = f2d.to(device)
                    # Forward an backward propagation
                    pred_theta, pred_phi, pred_dist, pred_omega = model(f2d)
                    pred_theta = soft(pred_theta)
                    pred_phi = soft(pred_phi)
                    pred_dist = soft(pred_dist)
                    pred_omega = soft(pred_omega)
                    theta = pred_theta.cpu().numpy().squeeze(0).transpose(1,2,0)
                    phi = pred_phi.cpu().numpy().squeeze(0).transpose(1,2,0)
                    dist = pred_dist.cpu().numpy().squeeze(0).transpose(1,2,0)
                    omega = pred_omega.cpu().numpy().squeeze(0).transpose(1,2,0)
                    #transform from dist to contact
                    dist2con(dist,outfile2)
                    ####
                    output_dict = dict(zip(['theta', 'phi', 'dist', 'omega'], [theta, phi, dist, omega]))
                    np.savez_compressed(outfile1, **output_dict)
                else:
                    print(prot+" featfile does not exists!")
            except Error:
                print("Error")

def main_single(prot, outfile, module_list):
    
    npzfiles = [outfile + "/N1_" + prot + i + ".npz" for i in module_list]
    #pzfniles = [outfile + "/" + prot + i + ".npz" for i in module_list]
    pred = np.array([])
    num_method = 0
    for npz_file in npzfiles:
        if os.path.exists(npz_file):
            npz = np.load(npz_file,allow_pickle=True)
            dist_org = npz['dist']
            num_method += 1
            if pred.shape[0] == 0:
                pred = dist_org
            else:
                pred = pred + dist_org

    if pred.shape[0] != 0:
        pred = pred / num_method
        n2_outfile = outfile + "/" + prot + ".map_std"
        dist2con(pred, n2_outfile) 
    else:
        print("not find predict result!")


def main(args):
    module_list = args.mid # "12345"
    print("method is N1_M" + module_list)
    # single method predict 
    for id in module_list:
       predict(args, id)
    
    # gather single method predict 
    chainlist = args.target
    with open(chainlist,"r") as chainfile:
        protlist = chainfile.readlines()
    protlist = [prot.strip() for prot in protlist]
    #parm_list = [(args.output, prot, module_list) for prot in protlist]
    print(protlist)
    with Pool(20) as p:
        p.map(partial(main_single,outfile=args.output, module_list=module_list), protlist)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('-m', '--model', type=str, default="../models", help='directory containing model ckpt files')
    parser.add_argument('-t', '--target', type=str, default="../chain", help='file containging target protein chain name')
    parser.add_argument('-f', '--feat', type=str, default="../input", help='directory containing feature files (npz)')
    #parser.add_argument('--cmap', type=str, default="../results", help='directory to save contact map files')
    parser.add_argument('-o', '--output', type=str, default="../results", help='directory to save output files')
    parser.add_argument('-g', '--gpu', type=str, default="0", help='use which GPU')
    parser.add_argument('--mid', type=str, default="12345", help='predict method 1 2 3 4 5')
    args = parser.parse_args()
    main(args)


