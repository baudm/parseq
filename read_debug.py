#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from modulefinder import packagePathMap
import os
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import normalize
from scipy.special import softmax

import torch
import hydra
from hydra.utils import instantiate
from hydra import compose, initialize
from omegaconf import OmegaConf

from strhub.data.module_debug import SceneTextDataModule
from strhub.models.utils import parse_model_args, init_dir

import warnings
warnings.filterwarnings('ignore')


@torch.inference_mode()
def main():
    def str2bool(x):
        return x.lower() == 'true'
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--images', nargs='+', required=True, help='Images to read')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')
    
    # if pretrained:
    #     try:
    #         url = _WEIGHTS_URL[experiment]
    #     except KeyError:
    #         raise InvalidModelError("No pretrained weights found for '{}'".format(experiment)) from None
    #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location='cpu', check_hash=True)
    #     model.load_state_dict(checkpoint)
    
    ckpt_split = args.checkpoint.split('/')
    exp_dir = '/'.join(ckpt_split[:ckpt_split.index('checkpoints')])
    initialize(config_path=f'{exp_dir}/config', version_base='1.2')
    cfg = compose(config_name='config')
    cfg.model._target_ = cfg.model._target_.replace('system', 'system_debug')
    for k, v in kwargs.items():
        setattr(cfg.model, k, v)
    
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    model.eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    
    hparams = model.hparams
    
    debug_dir = f'{exp_dir}/debug'
    init_dir(f'{debug_dir}/demo_images')
    
    for fname in args.images:
        basename = os.path.basename(fname)
        image_save_path = f'{debug_dir}/demo_images/{basename}'
        
        # Load image and prepare for input
        image = Image.open(fname).convert('RGB')
        image.save(image_save_path)
        image_t = img_transform(image).unsqueeze(0).to(args.device)

        logits, agg = model(image_t)
        p = logits.softmax(-1)
        pred, p_seq = model.tokenizer.decode(p)
        
        ## prediction
        # visualize_char_probs(pred, p, model, image_save_path)
        
        ## embeddings
        # visualize_head_self_sim(model, image_save_path)
        # visualize_pe_self_sim(pred, model, image_save_path)
        # visualize_text_embed_sim_with_head(model, image_save_path)
        # visualize_tsne(model, image_save_path)
        
        ## forward pass
        # visualize_sim_with_pe(model.pos_queries, ['*'*25], model, image_save_path, sim_scale=1.0)
        # visualize_sim_with_pe(agg.res_pt_1, pred, model, image_save_path, sim_scale=1.0)
        # for attr in ['main_pt_1', 'main_pt_2', 'main_pt_3', 'main_pt_4', 'res_pt_1', 'res_pt_2', 'res_pt_3']:
        # for attr in ['content']:
        #     visualize_sim_with_head(attr, agg, pred, model, image_save_path, sim_scale=2.0)
        
        ## attention
        # visualize_self_attn(pred, agg.sa_weights, image_save_path)
        # visualize_self_attn_VLP(pred, agg.sa_weights_dec, hparams, image, image_save_path, Q='V', K='V', tag=f'_dec')
        visualize_self_attn_VLP(pred, agg.sa_weights_ref, hparams, image, image_save_path, Q='P', K='P', tag=f'_ref')
        # visualize_cross_attn(agg.ca_weights, hparams, image, image_save_path)
        # visualize_sim_with_memory(agg.res_pt_2, agg.memory, image, image_save_path)
        
        print(f'{fname}: {pred[0]}')


def save_heatmap(data, rows, cols, title, save_path, sim_scale,
                 figsize=(15, 15), dpi=96, vmin=0, vmax=1,
                 annot=False, annot_size=10, tick_size=None,
                 linewidths=0, labelsize=None, x_rot=0, y_rot=0,
                 rects=None):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data *= sim_scale
    df = pd.DataFrame(data, index=rows, columns=cols)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.title(title)
    ax = plt.gca()
    # ax_pos = [0.15, 0.01, 0.84, 0.84]
    # ax.set_position(ax_pos)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="5%")
    sa = sns.heatmap(df,
                    vmin=vmin,
                    vmax=vmax,
                    annot=annot,
                    fmt='.2f',
                    annot_kws={'size': annot_size},
                    ax=ax,
                    cbar_ax=cax,
                    cbar=True,
                    linewidths=linewidths,
                    )
    cbar = sa.collections[0].colorbar
    if labelsize is not None:
        cbar.ax.tick_params(labelsize=labelsize)
    if rects is not None:
        for rect in rects:
            sa.add_patch(rect)
    sa.xaxis.tick_top()
    if tick_size is not None:
        sa.set_xticklabels(sa.get_xmajorticklabels(), fontsize=tick_size, rotation=x_rot)
        sa.set_yticklabels(sa.get_ymajorticklabels(), fontsize=tick_size, rotation=y_rot)
    else:
        sa.set_xticklabels(sa.get_xmajorticklabels(), rotation=x_rot)
        sa.set_yticklabels(sa.get_ymajorticklabels(), rotation=y_rot)
    plt.savefig(save_path)
    plt.close(fig)
    

def save_blended_heatmap(data, image, save_path, alpha=0.7):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    cm = plt.get_cmap('jet')
    attn = data
    attn = (attn - attn.min()) / (attn.max() - attn.min())
    attn = np.clip(attn, 0.0, 1.0)
    attn = cm(attn)
    attn = Image.fromarray((attn * 255).astype(np.uint8)).convert('RGB')
    attn = attn.resize(image.size)
    blend = Image.blend(image, attn, alpha=alpha)
    blend.save(save_path)



def visualize_self_attn_VLP(pred, sa_weights, hparams, image, image_save_path, tag='', Q='VLP', K='VLP', sim_scale=1.0):
    """
    Self-attn visualization of multi-modal Transformer.
    
    Args:
        Q : Query. e.g. 'V', 'L', 'P'
        K : Key. e.g. 'V', 'L', 'P'
    """
    if sa_weights is None: return
    filename_path, ext = os.path.splitext(image_save_path)
    pred = list(pred[0])
    vis_size = [a // b for (a, b) in zip(hparams.img_size, hparams.patch_size)]
    L_V = vis_size[0] * vis_size[1]
    L_L = L_P = hparams.max_label_length + 1
    L_T = L_V + L_L + L_P
    assert sa_weights.shape[-1] == L_T + 1 # +1 for dummy token
    rows = list(range(L_T + 1))
    for i in range(L_L + L_P):
        rows[L_V + i] = '[P]'
    for i in range(len(pred) + 1):
        rows[L_V + i] = (['[B]'] + pred)[i]
    for i in range (len(pred) + 1):
        rows[L_V + L_L + i] = (pred + ['[E]'])[i]
    rows[-1] = '[D]'
    rows_V = rows[:L_V]
    rows_L = rows[L_V:L_V + L_L]
    rows_P = rows[L_V + L_L:L_V + L_L + L_P]
    V_ind = list(range(L_V))
    L_ind = list(range(L_V, L_V + L_L))
    P_ind = list(range(L_V + L_L, L_V + L_L + L_P))
    rows, cols, row_ind, col_ind = [], [], [], []
    if 'V' in Q:
        rows.extend(rows_V)
        row_ind.extend(V_ind)
    if 'L' in Q:
        rows.extend(rows_L)
        row_ind.extend(L_ind)
    if 'P' in Q:
        rows.extend(rows_P)
        row_ind.extend(P_ind)
    if 'V' in K:
        cols.extend(rows_V)
        col_ind.extend(V_ind)
    if 'L' in K:
        cols.extend(rows_L)
        col_ind.extend(L_ind)
    if 'P' in K:
        cols.extend(rows_P)
        col_ind.extend(P_ind)
    
    if Q == K == 'VLP':
        rects = []
        for x, y, w, h in [(0, 0, L_V, L_V), (L_V, 0, L_L, L_V), (L_V + L_L, 0, L_P, L_V),
         (0, L_V, L_V, L_L), (L_V, L_V, L_L, L_L), (L_V + L_L, L_V, L_P, L_L),
         (0, L_V + L_L, L_V, L_P), (L_V, L_V + L_L, L_L, L_P), (L_V + L_L, L_V + L_L, L_P, L_P)]:
            rects.append(patches.Rectangle((x, y,), w, h, edgecolor='green', facecolor='none'))
        t = len(pred)
        for x, y, w, h in [(L_V, L_V, t, t), (L_V + L_L, L_V, t, t), (L_V, L_V + L_L, t, t), (L_V + L_L, L_V + L_L, t, t)]:
            rects.append(patches.Rectangle((x, y,), w, h, edgecolor='white', facecolor='none', linewidth=0.7))
        save_heatmap(sa_weights[-1][row_ind, :][:, col_ind], rows, cols, f'{Q}-{K}', f'{filename_path}_sa{tag}{ext}', sim_scale, rects=rects)
    elif Q + K in ['LL', 'LP', 'PL', 'PP']:
        if 'dec' in tag:
            for t, sa_weights_t in enumerate(sa_weights):
                tag_t = f'{tag}_{t:02d}'
                sa_weights_t = sa_weights_t[row_ind, :][:, col_ind].detach().cpu().numpy()
                sa_weights_t_temp = np.zeros_like(sa_weights_t)
                sa_weights_t_temp[:t + 1, :t + 1] = sa_weights_t[:t + 1, :t + 1]
                sa_weights_t = sa_weights_t_temp
                rects = [patches.Rectangle((0, 0,), t + 1, t + 1, edgecolor='white', facecolor='none')]
                save_heatmap(sa_weights_t, rows, cols, f'{Q}-{K}', f'{filename_path}_sa{tag_t}{ext}', sim_scale, rects=rects, annot=True)
        elif 'ref' in tag:
            t = len(pred)
            sa_weights_t = sa_weights[0][row_ind, :][:, col_ind].detach().cpu().numpy()
            sa_weights_t_temp = np.zeros_like(sa_weights_t)
            sa_weights_t_temp[:t + 1, :t + 1] = sa_weights_t[:t + 1, :t + 1]
            sa_weights_t = sa_weights_t_temp
            rects = [patches.Rectangle((0, 0,), t + 1, t + 1, edgecolor='white', facecolor='none')]
            save_heatmap(sa_weights_t, rows, cols, f'{Q}-{K}', f'{filename_path}_sa{tag}{ext}', sim_scale, rects=rects, annot=True)
        else:
            raise NotImplementedError
    elif Q + K in ['PV', 'LV']:
        if 'dec' in tag:
            for t, sa_weights_t in enumerate(sa_weights):
                tag_t = f'{tag}_{t:02d}'
                save_path = f'{filename_path}_sa{tag_t}{ext}'
                sa_weights_t = sa_weights_t[row_ind, :][:, col_ind]
                sa_weights_t = sa_weights_t[t]
                sa_weights_t = sa_weights_t.view(*vis_size)
                save_blended_heatmap(sa_weights_t, image, save_path, tag_t)
        elif 'ref' in tag:
            t = len(pred)
            save_path = f'{filename_path}_sa{tag_t}{ext}'
            sa_weights_t = sa_weights[0][row_ind, :][:, col_ind]
            sa_weights_t = sa_weights_t[t]
            sa_weights_t = sa_weights_t.view(*vis_size)
            save_blended_heatmap(sa_weights_t, image, save_path, tag)
        else:
            raise NotImplementedError
    elif Q + K in ['VV']:
        sa_weights = sa_weights[-1]
        sa_weights = sa_weights[row_ind, :][:, col_ind]
        for pix in range(sa_weights.shape[0]):
            tag_t = f'{tag}_{pix:02d}'
            save_path = f'{filename_path}_sa{tag_t}{ext}'
            sa_weights_t = sa_weights[pix]
            sa_weights_t = sa_weights_t.view(*vis_size)
            attn = sa_weights_t.detach().cpu().numpy()
            attn = (attn - attn.min()) / (attn.max() - attn.min())
            attn = np.clip(attn, 0.0, 1.0)
            cm = plt.get_cmap('jet')
            attn = cm(attn)
            attn[pix // vis_size[1], pix % vis_size[1]] = [1, 0, 1, 1] # query pixel is magenta
            attn = Image.fromarray((attn * 255).astype(np.uint8)).convert('RGB')
            attn = attn.resize(image.size, resample=Image.NEAREST)
            blend = Image.blend(image, attn, alpha=0.5)
            blend.save(save_path)
    elif Q + K in ['VP', 'VL']:
        raise NotImplementedError
    else:
        raise NotImplementedError
        

def visualize_self_attn(pred, sa_weights, image_save_path, tag=''):
    if sa_weights is None: return
    filename_path, ext = os.path.splitext(image_save_path)
    # cols = ['[B]'] + list(pred[0])
    # rows = ['[B]'] + list(pred[0])
    # rows = list(pred[0]) + ['[E]']
    rows = list(range(sa_weights.shape[0]))
    cols = list(range(sa_weights.shape[1]))
    
    df = pd.DataFrame(sa_weights.detach().cpu().numpy(), index=rows, columns=cols)
    plt.figure(figsize=(15, 15), dpi=96)
    annot_size = 20
    tick_size = 20
    labelsize = 20
    save_path = f'{filename_path}_sa{tag}{ext}'
    ax = plt.gca()
    # ax_pos = [0.15, 0.01, 0.84, 0.84]
    # ax.set_position(ax_pos)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="5%")
    sa = sns.heatmap(df,
                    vmin=0,
                    vmax=1,
                    # annot=True,
                    fmt='.2f',
                    annot_kws={'size': annot_size},
                    ax=ax,
                    cbar_ax=cax,
                    cbar=True,
                    # linewidths=1,
                    )
    cbar = sa.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=labelsize)
    # sa.xaxis.tick_top()
    sa.set_xticklabels(sa.get_xmajorticklabels(), rotation=90)
    # sa.set_xticklabels(sa.get_xmajorticklabels(), fontsize=tick_size, rotation=0)
    # sa.set_yticklabels(sa.get_ymajorticklabels(), fontsize=tick_size, rotation=0)
    plt.savefig(save_path); plt.clf()
    
    
def visualize_tsne(model, image_save_path):
    filename_path, ext = os.path.splitext(image_save_path)
    from sklearn.manifold import TSNE
    head = model.head.weight.detach().cpu().numpy()
    charset_train = model.hparams.charset_train
    rows = ['[E]'] + list(charset_train)
    cols = ['x', 'y']
    tsne = TSNE(n_components=2).fit_transform(head)
    tsne = pd.DataFrame(tsne, index=rows, columns=cols)
    tsne['char'] = rows
    fig = plt.figure(figsize=(30, 30), dpi=300)
    sc = sns.scatterplot(data=tsne, x='x', y='y', hue='char', style='char')
    ax = plt.gca()
    for _, row in tsne.iterrows():
        ax.text(row['x'] + .02, row['y'], row['char'])
    save_path = f'{filename_path}_sc{ext}'
    plt.savefig(save_path)
    plt.close(fig)


def visualize_char_probs(pred, p, model, image_save_path):
    filename_path, ext = os.path.splitext(image_save_path)
    rows = pred = list(pred[0]) + ['[E]']
    p = p[0].detach().cpu().numpy()[:len(pred), :] # probs up to [E], [seq_len + 1, len(charset_train) - 2]
    charset_train = model.hparams.charset_train
    cols = ['[E]'] + list(charset_train)
    save_path = f'{filename_path}_p{ext}'
    save_heatmap(p, rows, cols, '', save_path, 1.0, figsize=(30, len(rows)), annot=True, annot_size=10, tick_size=15, labelsize=15, linewidths=1)

def visualize_similarity(target, source, rows, cols, image_save_path, sim_scale=1.0, annot=False, tag=''):
    filename_path, ext = os.path.splitext(image_save_path)
    target = normalize(target)
    source = normalize(source)
    similarity_mtx = target @  source.T
    save_heatmap(similarity_mtx, rows, cols, '', f'{filename_path}_sim{tag}{ext}', sim_scale, annot=annot, annot_size=10, tick_size=10, labelsize=10)
 

def visualize_head_self_sim(model, image_save_path):
    head = model.head.weight.detach().cpu().numpy()
    charset_train = model.hparams.charset_train
    rows = cols = (['[E]'] + list(charset_train) + ['[B]', '[P]'])[:head.shape[0]]
    visualize_similarity(head, head, rows, cols, image_save_path) 


def visualize_sim_with_head(attr, agg, pred, model, image_save_path, sim_scale=1.0):
    head = model.head.weight.detach().cpu().numpy()
    charset_train = model.hparams.charset_train
    if len(head) == 95:
        cols = ['[E]'] + list(charset_train)
    else:
        cols = ['[E]'] + list(charset_train) + ['[B]', '[P]']
    if attr == 'content':
        rows = ['[B]'] + list(pred[0])
    else:
        rows = list(pred[0]) + ['[E]']
    target = getattr(agg, attr)
    target = target.detach().cpu().numpy()[0]
    visualize_similarity(target, head, rows, cols, image_save_path, sim_scale=sim_scale, tag='_' + attr)


def visualize_pe_self_sim(pred, model, image_save_path, sim_scale=1.0):
    pred = list(pred[0]) + ['[E]']
    # pos_queries = model.pos_queries.detach().cpu().numpy()[0][:len(pred), :]
    pos_queries = model.pos_embed.detach().cpu().numpy()[0][:len(pred), :]
    rows = cols = list(range(1, len(pred) + 1))
    visualize_similarity(pos_queries, pos_queries, rows, cols, image_save_path, sim_scale, annot=True)
    
def visualize_sim_with_pe(target, pred, model, image_save_path, sim_scale=1.0):
    rows = pred = list(pred[0]) + ['[E]']
    pos_queries = model.pos_queries.detach().cpu().numpy()[0][:len(pred), :]
    target = target.detach().cpu().numpy()[0][:len(pred), :]
    cols = list(range(1, len(pred) + 1))
    visualize_similarity(pos_queries, pos_queries, rows, cols, image_save_path, sim_scale, annot=True)
    
    
def visualize_text_embed_sim_with_head(model, image_save_path): 
    text_embed = model.text_embed.embedding.weight.detach().cpu().numpy() # [charset_size, embed_dim]
    head = model.head.weight.detach().cpu().numpy()
    charset_train = model.hparams.charset_train
    rows = cols = (['[E]'] + list(charset_train) + ['[B]', '[P]'])[:head.shape[0]]
    visualize_similarity(text_embed, head, rows, cols, image_save_path)
            
   
def visualize_cross_attn(ca_weights, hparams, image, image_save_path, tag=''):
    filename_path, ext = os.path.splitext(image_save_path)
    if ca_weights is None: return
    vis_size = [a // b for (a, b) in zip(hparams.img_size, hparams.patch_size)]
    ca_weights = ca_weights.view(-1, vis_size[0], vis_size[1])
    ca_weights = ca_weights.detach().cpu().numpy()
    
    for i, attn in enumerate(ca_weights):
        save_path = f'{filename_path}_ca{tag}_{i:02d}{ext}'
        save_blended_heatmap(attn, image, save_path, alpha=0.8)
    

def visualize_sim_with_memory(target, memory, image, image_save_path):
    filename_path, ext = os.path.splitext(image_save_path)
    cm = plt.get_cmap('jet')
    memory = memory.view(-1, 384).detach().cpu().numpy()
    target = target.view(-1, 384).detach().cpu().numpy()
    seq_sim_mtx = target @ memory.T
    for i, sim_mtx in enumerate(seq_sim_mtx):
        save_path = f'{filename_path}_sm_{i:02d}{ext}'
        attn = softmax(sim_mtx)
        save_blended_heatmap(attn, image, save_path)

if __name__ == '__main__':
    main()
