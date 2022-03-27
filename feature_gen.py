import argparse
from pyexpat import features
import numpy as np
from os.path import join
import os
import parse_feature
import string, re
import torch
import torch.nn.functional as F

def d(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'
def msa2pssm(msa1hot, w):
    beff = w.sum()
    f_i = (w[:, None, None] * msa1hot).sum(dim=0) / beff + 1e-9
    h_i = (-f_i * torch.log(f_i)).sum(dim=1)
    return torch.cat((f_i, h_i[:, None]), dim=1)

# reweight MSA based on cutoff
def reweight(msa1hot, cutoff):
    id_min = msa1hot.shape[1] * cutoff
    id_mtx = torch.einsum('ikl,jkl->ij', msa1hot, msa1hot)
    id_mask = id_mtx > id_min
    w = 1. / id_mask.float().sum(dim=-1)
    return w

# shrunk covariance inversion
def fast_dca(msa1hot, weights, penalty = 4.5):
    device = msa1hot.device
    nr, nc, ns = msa1hot.shape
    x = msa1hot.view(nr, -1)
    num_points = weights.sum() - torch.sqrt(weights.mean())

    mean = (x * weights[:, None]).sum(dim=0, keepdims=True) / num_points
    x = (x - mean) * torch.sqrt(weights[:, None])

    cov = (x.t() @ x) / num_points
    cov_reg = cov + torch.eye(nc * ns).to(device) * penalty / torch.sqrt(weights.sum())

    inv_cov = torch.inverse(cov_reg)
    x1 = inv_cov.view(nc, ns, nc, ns)
    x2 = x1.transpose(1, 2).contiguous()
    features = x2.reshape(nc, nc, ns * ns)

    x3 = torch.sqrt((x1[:, :-1, :, :-1] ** 2).sum(dim=(1, 3))) * (1 - torch.eye(nc).to(device))
    apc = x3.sum(dim=0, keepdims=True) * x3.sum(dim=1, keepdims=True) / x3.sum()
    contacts = (x3 - apc) * (1 - torch.eye(nc).to(device))
    return torch.cat((features, contacts[:, :, None]), dim=2)

def parse_a3m(filename, limit=70000):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase + '*'))

    seq_len = len(open(filename, "r").readlines()[1]) - 1
    # read file line by line
    count = 0
    for line in open(filename, "r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            line = line.rstrip().translate(table)
            if len(line) != seq_len:
                continue
            seqs.append(line.rstrip().translate(table))
            count += 1
            if count >= limit:
                break

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    return msa

def collect_features(a3m, wmin=0.8, ns=21):
    #Huiling a3m = torch.from_numpy(parse_a3m(msa_file)).long()
    #a3m = torch.from_numpy(a3m).long()
    nrow, ncol = a3m.shape
    msa1hot = F.one_hot(a3m, ns).float().to(d())
    w = reweight(msa1hot, wmin).float().to(d())

    # 1d sequence

    f1d_seq = msa1hot[0, :, :20].float()
    f1d_pssm = msa2pssm(msa1hot, w)

    f1d = torch.cat((f1d_seq, f1d_pssm), dim=1)
    f1d = f1d[None, :, :].reshape((1, ncol, 42))

    # 2d sequence

    f2d_dca = fast_dca(msa1hot, w) if nrow > 1 else torch.zeros((ncol, ncol, 442)).float().to(d())
    f2d_dca = f2d_dca[None, :, :, :]

    f2d = torch.cat((
        f1d[:, :, None, :].repeat(1, 1, ncol, 1),
        f1d[:, None, :, :].repeat(1, ncol, 1, 1),
        f2d_dca
    ), dim=-1)

    f2d = f2d.view(1, ncol, ncol, 442 + 2*42)
    return f2d.permute((0, 3, 2, 1))

def generate_feature_m1(target, input_dir, output_dir):
    a3m_path = input_dir + "/" + target
    if os.path.exists(a3m_path + "/"+ target + ".a3m"):
        a3mfile = a3m_path + "/"  +target + ".a3m"
    elif os.path.exists(a3m_path + "/" + target + ".aln"):
        a3mfile = a3m_path + "/" + target + ".aln"
    elif os.path.exists(a3m_path + "/"+target + ".fasta"):
        a3mfile = a3m_path + "/" + target + ".fasta"
    else:
        raise Exception("a3mfile does not exist!")
    
    if os.path.exists(a3mfile):
        msa = parse_a3m(a3mfile)
        msa = torch.from_numpy(msa).to(torch.int64)
        msa = torch.squeeze(msa, dim=0)
        feat = collect_features(msa)
        print("feature shape:%s, feature type:%s"%(feat.shape,feat.dtype))
        #f2d = torch.tensor(load_features(featfile, prot))
        feature_m_file = "%s_features_m1"%(target)
        np.savez(join(output_dir, feature_m_file), features=feat)
        print("feature for M1 2 saved.")

def generate_feature_m3(target, input_dir, output_dir):

    features = {}

    fasta_file = "%s/%s.fa"%(target, target)
    sequence = parse_feature.get_seq(join(input_dir,fasta_file))
    #features['sequence'] = sequence
    feat_seq = sequence
    #print(feat_seq)
    Nseq=len(feat_seq)

    pssm_file = "%s/%s.pssm"%(target,  target)
    pssm = parse_feature.pssm_parser(join(input_dir,pssm_file))
    #features['pssm'] = pssm
    feat_pssm = pssm
    #print(feat_pssm.shape)
 
    hmm_file = "%s/%s.hhm"%(target,  target)
    hmm = parse_feature.hmm_profile_parser(join(input_dir,hmm_file), sequence)
    #features['hmm'] = hmm
    feat_hmm = hmm
    #print(feat_hmm.shape)

    spot1d_file = "%s/%s.spd33"%(target, target)
    spot1d = parse_feature.spot1d_parser(join(input_dir,spot1d_file))
    #features['spot1d'] = spot1d
    feat_spot1d = spot1d
    #print(feat_spot1d.shape)

    onehot_encoding = parse_feature.get_onehotencoding(sequence)
    #features['onehot'] = onehot_encoding
    feat_onehot = onehot_encoding
    #print(feat_onehot.shape)

    ccmpred_file = "%s/%s.ccmpred"%(target, target)
    ccmpred = parse_feature.ccmpred_parser(join(input_dir,ccmpred_file))
    #features['ccmpred'] = ccmpred
    feat_ccmpred = ccmpred
    #print(feat_ccmpred.shape)
 
    mutualinfo_file = "%s/%s.mutualinfo"%(target, target)
    mutualinfo = parse_feature.mutualinfo_parser(join(input_dir,mutualinfo_file))
    #features['mi'] = mutualinfo
    feat_mi = mutualinfo
    #print(feat_mi.shape)

    potential_file = "%s/%s.potential"%(target, target)
    potential = parse_feature.potential_parser(join(input_dir,potential_file))
    #features['potential'] = potential
    feat_potential = potential
    #print(feat_potential.shape)
    f1d = np.concatenate((feat_pssm.T,feat_hmm.T,feat_spot1d.T,feat_onehot.T),axis=1)
    #print(f1d.shape)
    f1d = f1d[None, :, :].reshape((1, f1d.shape[0], f1d.shape[1]))
    f2d = np.concatenate((feat_ccmpred.transpose(1,2,0),feat_mi.transpose(1,2,0),feat_potential.transpose(1,2,0)),axis=2)
    f2d = f2d[None, :, :, :]
    feat = np.concatenate((np.tile(f1d[:, :, None, :],(1, 1, Nseq, 1)),
                           np.tile(f1d[:, None, :, :],(1, Nseq, 1, 1)),
                           f2d
                           ),axis=-1)

    feat = feat.astype(np.float32)
    feat = feat.transpose(0,3,1,2)
    print("feature shape:%s, feature type:%s"%(feat.shape,feat.dtype))
    
    # save feature to file
    feature_file = "%s_features_m3"%(target)
    np.savez(join(output_dir,feature_file), features=feat)
    print("feature for M3 4 5 saved.")


def main(args):

    target = args.target
    input_dir = args.input_dir
    output_dir = args.output_dir
    # evalue=''
    
    with open(target,"r") as target_file:
        target_list = target_file.readlines()
    target_list = [target.strip() for target in target_list]
    print(target_list)
    for target in target_list:
        feature_path = join(output_dir, target)
        print(feature_path)
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        # M1 2 feature
        generate_feature_m1(target, input_dir, feature_path)
        # M3 4 5feature
        generate_feature_m3(target, input_dir, feature_path)
        
        print("Features computed for target %s"%(target))
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('-t', '--target', type=str, default="../chain", help='target protein name')
    parser.add_argument('-f', '--input_dir', type=str, default="../input", help='directory containing input files')
    parser.add_argument('-o', '--output_dir', type=str, default="../feature", help='directory containing output files')
    args = parser.parse_args()
    main(args)

