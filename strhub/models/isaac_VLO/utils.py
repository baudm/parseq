import torch
import torch.nn.functional as F

class AttentionMask:
    def __init__(self, max_label_length, QK, hparams):
        self.max_label_length = max_label_length
        self.QK = QK
        self.hparams = hparams
        
    def get_attn_mask(self, img_size, patch_size, refine_layer:bool=False):
        """Generates attention mask for the multi-modal transformer layers.
        
        Args:
            refine_layer: Whether or not the layer is used for refinement (as opposed to initial text prediction).
                When False, since information leak to future time steps are not allowed,
                - visual tokens cannot attend to language or ordinal tokens
                - causal mask is applied between language and ordinal tokens
                When True, it assumes an initial text prediction (up to <eos>) is already made.
                - full attention between visual, langauge and ordinal tokens is applied.
        """
        L_V = int(img_size[0] * img_size[1] / (patch_size[0] * patch_size[1]))
        L_L = L_O = self.max_label_length + 1 # +1 for eos
        L_T = L_V + L_L + L_O
        def full_attn(h, w=None):
            w = w if w is not None else h
            return torch.zeros((h, w))
        def zero_attn(h, w=None):
            w = w if w is not None else h
            return torch.full((h, w), float('-inf'))
        def causal_attn(h, w=None, include_self=True):
            w = w if w is not None else h
            diagonal = 1 if include_self == True else 0
            return torch.triu(torch.full((h, w), float('-inf')), diagonal)
        def diag_attn(h, w=None):
            w = w if w is not None else h
            triu = torch.triu(torch.full((h, w), float('-inf')), 1)
            tril = torch.tril(torch.full((h, w), float('-inf')), -1)
            return triu + tril
        def diag_mask(h, w=None, diagonal=0):
            w = w if w is not None else h
            base = torch.full((h, w), 1.0)
            triu = torch.triu(torch.full((h, w), -1.0), diagonal + 1)
            tril = torch.tril(torch.full((h, w), -1.0), diagonal - 1)
            mask = base + triu + tril
            mask = torch.zeros((h, w)).masked_fill(mask.type(torch.bool), float('-inf'))
            return mask
        
        # query : V
        QK_V = self.QK[0]
        if 'V' in QK_V:
            attn_VV = full_attn(L_V)
        else:
            attn_VV = zero_attn(L_V)
        if 'L' in QK_V and not not refine_layer:
            # VL attention is not allowed in base layer, due to information leak from future time steps
            attn_VL = full_attn(L_V, L_L)
        else:
            attn_VL = zero_attn(L_V, L_L)
        if 'P' in QK_V and not not refine_layer:
            # VP attention is not allowed in base layer, due to information leak from future time steps
            attn_VP = full_attn(L_V, L_O)
        else:
            attn_VP = zero_attn(L_V, L_O)
        attn_V = torch.cat((attn_VV, attn_VL, attn_VP), dim=1)
        
        # query : L
        QK_L = self.QK[1]
        if 'V' in QK_L:
            attn_LV = full_attn(L_L, L_V)
        else:
            attn_LV = zero_attn(L_L, L_V)
        if 'L' in QK_L:
            if not refine_layer:
                attn_LL = causal_attn(L_L)
            else:
                attn_LL = full_attn(L_L)
        else:
            attn_LL = zero_attn(L_L)
        if 'P' in QK_L:
            if not refine_layer:
                attn_LP = causal_attn(L_L, L_O)
            else:
                attn_LP = full_attn(L_L, L_O)
        else:
            attn_LP = zero_attn(L_L, L_O)
        attn_L = torch.cat((attn_LV, attn_LL, attn_LP), dim=1)
        
        # query : P
        QK_P = self.QK[2]
        if 'V' in QK_P:
            attn_PV = full_attn(L_O, L_V)
        else:
            attn_PV = zero_attn(L_O, L_V)
        if 'L' in QK_P:
            if not refine_layer:
                attn_PL = causal_attn(L_O, L_L)
            else:
                attn_PL = full_attn(L_O, L_L)
        else:
            attn_PL = zero_attn(L_O, L_L)
        if 'P' in QK_P:
            if not refine_layer:
                attn_PP = causal_attn(L_O)
            else:
                attn_PP = full_attn(L_O)
        else:
            attn_PP = zero_attn(L_O)
        attn_P = torch.cat((attn_PV, attn_PL, attn_PP), dim=1)
        
        attn_mask = torch.cat((attn_V, attn_L, attn_P), dim=0)
        attn_mask = self.add_dummy_attn(attn_mask)
        
        return attn_mask

    def add_dummy_attn(self, attn_mask):
        """ Add attention to dummy token(extra fixed zero token),
        which is appended to the end of the concatenated tokens, to get around the
        gradient error caused by all keys being masked. When all keys are masked,
        attention to the dummy token is enabled.
        """
        attn_mask = F.pad(attn_mask, (0, 0, 0, 1), 'constant', float('-inf'))
        attn_mask = F.pad(attn_mask, (0, 1), 'constant', 0)
        for i, row in enumerate(attn_mask):
            if torch.any(row[:-1] != float('-inf')):
                attn_mask[i, -1] = float('-inf')
        return attn_mask

    def visualize_attn_mask(self, attn_mask, refine_layer:bool=False):
        import seaborn as sns
        import pandas as pd
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        vis_size = [a // b for (a, b) in zip(self.hparams.img_size, self.hparams.patch_size)]
        L_V = vis_size[0] * vis_size[1]
        L_L = L_O = self.max_label_length + 1
        L_T = L_V + L_L + L_O
        win = attn_mask.shape[0]
        df = pd.DataFrame(torch.where(attn_mask == 0, 1, 0).numpy()[-win:, -win:], index=list(range(win)), columns=list(range(win)))
        s = 1.0
        plt.figure(figsize=(30 * s, 30 * s), dpi=300)
        annot_size = 10 * s
        tick_size = 5 * s
        labelsize = 15 * s
        if refine_layer:
            save_path = f'./attn_refine.png'
        else:
            save_path = f'./attn.png'
        ax = plt.gca()
        # ax_pos = [0.15, 0.01, 0.84, 0.84]
        # ax.set_position(ax_pos)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="5%")
        sa = sns.heatmap(df,
                        vmin=0,
                        vmax=1,
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
        rects = []
        for x, y, w, h in [(0, 0, L_V, L_V), (L_V, 0, L_L, L_V), (L_V + L_L, 0, L_O, L_V), (L_T, 0, 1, L_V),
            (0, L_V, L_V, L_L), (L_V, L_V, L_L, L_L), (L_V + L_L, L_V, L_O, L_L), (L_T, L_V, 1, L_L),
            (0, L_V + L_L, L_V, L_O), (L_V, L_V + L_L, L_L, L_O), (L_V + L_L, L_V + L_L, L_O, L_O), (L_T, L_V + L_L, 1, L_O),
            (0, L_T, L_V, 1), (L_V, L_T, L_L, 1), (L_V + L_L, L_T, L_O, 1), (L_T, L_T, 1, 1),
            ]:
            rects.append(patches.Rectangle((x, y,), w, h, edgecolor='green', facecolor='none', linewidth=3))
        for rect in rects:
            sa.add_patch(rect)
        sa.xaxis.tick_top()
        sa.set_xticklabels(sa.get_xmajorticklabels(), fontsize=tick_size, rotation=0)
        sa.set_yticklabels(sa.get_ymajorticklabels(), fontsize=tick_size, rotation=0)
        plt.savefig(save_path); plt.clf()