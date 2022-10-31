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
from sklearn.preprocessing import normalize
from scipy.special import softmax

import torch
import hydra
from hydra.utils import instantiate
from hydra import compose, initialize
from omegaconf import OmegaConf

from strhub.data.module_debug import SceneTextDataModule
from strhub.models.utils import parse_model_args, init_dir



@torch.inference_mode()
def main():
    def str2bool(x):
        return x.lower() == 'true'
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--images', nargs='+', help='Images to read')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sa', type=str2bool, default=True, help='Output self-attention maps')
    parser.add_argument('--ca', type=str2bool, default=True, help='Output cross-attention maps')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')
    
    ckpt_split = args.checkpoint.split('/')
    exp_dir = '/'.join(ckpt_split[:ckpt_split.index('checkpoints')])
    debug_dir = f'{exp_dir}/debug'
    init_dir(f'{debug_dir}/demo_images')
    
    # if pretrained:
    #     try:
    #         url = _WEIGHTS_URL[experiment]
    #     except KeyError:
    #         raise InvalidModelError("No pretrained weights found for '{}'".format(experiment)) from None
    #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location='cpu', check_hash=True)
    #     model.load_state_dict(checkpoint)

    initialize(config_path=f'{exp_dir}/config', version_base='1.2')
    cfg = compose(config_name='config')
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    model.eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    
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
        
        # visualize_text_embed_sim_with_head(model, image_save_path)
        # visualize_head_self_sim(model, image_save_path)
        # visualize_sim_with_pe(model.pos_queries, ['*'*25], model, image_save_path, sim_scale=1.0)
        # visualize_sim_with_pe(agg.res_pt_1, pred, model, image_save_path, sim_scale=1.0)
        # for attr in ['main_pt_1', 'main_pt_2', 'main_pt_3', 'main_pt_4', 'res_pt_1', 'res_pt_2', 'res_pt_3']:
        # for attr in ['content']:
        #     visualize_sim_with_head(attr, agg, pred, model, image_save_path, sim_scale=2.0)
        # visualize_sim_with_memory(image, agg.res_pt_2, agg.memory, image_save_path)
        visualize_char_probs(pred, p, model, image_save_path)
        # visualize_attn(args, image, agg.sa_weights, agg.ca_weights, image_save_path)
        # visualize_self_attn(pred, agg.sa_weights, image_save_path)
        # visualize_cross_attn(image, agg.ca_weights, image_save_path)
        print(f'{fname}: {pred[0]}')
        

def visualize_head_self_sim(model, image_save_path):
    head = model.head.weight.detach().cpu().numpy()
    charset_train = model.hparams.charset_train
    # rows = cols = ['[E]'] + list(charset_train)
    rows = cols = ['[E]'] + list(charset_train) + ['[B]', '[P]']
    visualize_similarity(head, head, rows, cols, image_save_path)        


def visualize_char_probs(pred, p, model, image_save_path):
    filename_path, ext = os.path.splitext(image_save_path)
    rows = pred = list(pred[0]) + ['[E]']
    p = p[0].detach().cpu().numpy()[:len(pred), :] # probs up to [E], [seq_len + 1, len(charset_train) - 2]
    charset_train = model.hparams.charset_train
    cols = ['[E]'] + list(charset_train)
    df = pd.DataFrame(p, index=rows, columns=cols)
    s = 1.0
    plt.figure(figsize=(30 * s, len(rows) * s), dpi=300)
    annot_size = 10 * s
    tick_size = 15 * s
    labelsize = 15 * s
    save_path = f'{filename_path}_p{ext}'
    ax = plt.gca()
    # ax_pos = [0.15, 0.01, 0.84, 0.84]
    # ax.set_position(ax_pos)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="5%")
    sa = sns.heatmap(df,
                    # vmin=0,
                    # vmax=1,
                    # annot=True,
                    # fmt='.2f',
                    # annot_kws={'size': annot_size},
                    ax=ax,
                    cbar_ax=cax,
                    cbar=True,
                    linewidths=0.5,
                    )
    cbar = sa.collections[0].colorbar
    cbar.ax.tick_params(labelsize=labelsize)
    sa.xaxis.tick_top()
    sa.set_xticklabels(sa.get_xmajorticklabels(), fontsize=tick_size, rotation=0)
    sa.set_yticklabels(sa.get_ymajorticklabels(), fontsize=tick_size, rotation=0)
    plt.savefig(save_path); plt.clf()
    

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
    rows = ['[E]'] + list(charset_train) + ['[B]', '[P]']
    cols = ['[E]'] + list(charset_train) + ['[B]', '[P]']
    # cols = ['[E]'] + list(charset_train)
    visualize_similarity(text_embed, head, rows, cols, image_save_path)
            

def visualize_similarity(target, source, rows, cols, image_save_path, sim_scale=1.0, annot=False, tag=''):
    filename_path, ext = os.path.splitext(image_save_path)
    target = normalize(target)
    source = normalize(source)
    similarity_mtx = target @  source.T
    similarity_mtx *= sim_scale
    df = pd.DataFrame(similarity_mtx, index=rows, columns=cols) # [tgt x src]
    fig_scale = 1.0
    plt.figure(figsize=(min(len(cols), 30) * fig_scale, min(len(rows), 30) * fig_scale), dpi=300)
    annot_size = 10 * fig_scale
    tick_size = 10 * fig_scale
    labelsize = 10 * fig_scale
    save_path = f'{filename_path}_sim{tag}{ext}'
    ax = plt.gca()
    # ax_pos = [0.15, 0.01, 0.84, 0.84]
    # ax.set_position(ax_pos)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="5%")
    sa = sns.heatmap(df,
                    vmin=0,
                    vmax=1,
                    annot=annot,
                    fmt='.2f',
                    annot_kws={'size': annot_size},
                    ax=ax,
                    cbar_ax=cax,
                    cbar=True
                    )
    cbar = sa.collections[0].colorbar
    cbar.ax.tick_params(labelsize=labelsize)
    sa.xaxis.tick_top()
    sa.set_xticklabels(sa.get_xmajorticklabels(), fontsize=tick_size, rotation=0)
    sa.set_yticklabels(sa.get_ymajorticklabels(), fontsize=tick_size, rotation=0)
    plt.savefig(save_path); plt.close()
    
        
def visualize_attn(args, image, sa_weights, ca_weights, image_save_path):
    image.save(image_save_path)
    if args.sa:
        visualize_self_attn(sa_weights, image_save_path)
    if args.ca:
        visualize_cross_attn(image, ca_weights, image_save_path)
    
    
def visualize_self_attn(pred, sa_weights, image_save_path):
    if sa_weights is None: return
    pred = ['[B]'] + list(pred[0])
    filename_path, ext = os.path.splitext(image_save_path)
    seq_len = sa_weights.shape[0]
    cols = pred
    rows = list(range(1, seq_len + 1))
    df = pd.DataFrame(sa_weights.detach().cpu().numpy(), index=rows, columns=cols)
    s = 1.0
    plt.figure(figsize=(15 * s, 15 * s), dpi=300)
    annot_size = 20 * s
    tick_size = 20 * s
    labelsize = 20 * s
    save_path = f'{filename_path}_sa{ext}'
    ax = plt.gca()
    # ax_pos = [0.15, 0.01, 0.84, 0.84]
    # ax.set_position(ax_pos)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad="5%")
    sa = sns.heatmap(df,
                    vmin=0,
                    vmax=1,
                    annot=True,
                    fmt='.2f',
                    annot_kws={'size': annot_size},
                    ax=ax,
                    cbar_ax=cax,
                    cbar=True
                    )
    cbar = sa.collections[0].colorbar
    cbar.ax.tick_params(labelsize=labelsize)
    sa.xaxis.tick_top()
    sa.set_xticklabels(sa.get_xmajorticklabels(), fontsize=tick_size, rotation=0)
    sa.set_yticklabels(sa.get_ymajorticklabels(), fontsize=tick_size, rotation=0)
    plt.savefig(save_path); plt.clf()
    
    
def visualize_cross_attn(image, ca_weights, image_save_path):
    filename_path, ext = os.path.splitext(image_save_path)
    if ca_weights is None: return
    ca_weights = ca_weights.view(-1, 8, 16)
    ca_weights = ca_weights.detach().cpu().numpy()
    
    cm = plt.get_cmap('jet')
    for i, attn in enumerate(ca_weights):
        i += 1
        save_path = f'{filename_path}_ca_{i:02d}{ext}'
        # attn *= 10
        attn = (attn - attn.min()) / (attn.max() - attn.min())
        attn = np.clip(attn, 0.0, 1.0)
        attn = cm(attn)
        attn = Image.fromarray((attn * 255).astype(np.uint8)).convert('RGB')
        attn = attn.resize(image.size)
        blend = Image.blend(image, attn, alpha=0.8)
        blend.save(save_path)
    

def visualize_sim_with_memory(image, target, memory, image_save_path):
    filename_path, ext = os.path.splitext(image_save_path)
    cm = plt.get_cmap('jet')
    memory = memory.view(-1, 384).detach().cpu().numpy()
    target = target.view(-1, 384).detach().cpu().numpy()
    seq_sim_mtx = target @ memory.T
    for i, sim_mtx in enumerate(seq_sim_mtx):
        save_path = f'{filename_path}_sm_{i:02d}{ext}'
        attn = softmax(sim_mtx)
        attn = attn * 10
        # attn = (attn - attn.min()) / (attn.max() - attn.min())
        attn = np.clip(attn, 0.0, 1.0)
        attn = attn.reshape((8, 16))
        attn = cm(attn)
        attn = Image.fromarray((attn * 255).astype(np.uint8)).convert('RGB')
        attn = attn.resize(image.size)
        blend = Image.blend(image, attn, alpha=0.8)
        blend.save(save_path)

if __name__ == '__main__':
    main()
