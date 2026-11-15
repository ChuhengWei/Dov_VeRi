"""Shared protocol: data lists, feature extraction driver, metrics.

Every method uses THIS file for evaluation so the protocol is bit-identical across methods.
"""
import os, json, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_PATH = os.path.join(HERE, '..', 'split.json')

NOTE = ("NOTE: exactly one correct gallery entry per query at ID level, so AP = 1/rank "
        "and mAP is identical to mean reciprocal rank (MRR).")


class Protocol:
    def __init__(self, path=SPLIT_PATH):
        with open(path) as f:
            self.s = json.load(f)
        self.root = self.s['root']
        if not os.path.isabs(self.root):
            self.root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), self.root))
        self.root = os.environ.get('ROADCAM_ROOT', self.root)
        self.man = self.s['manifest']
        self.ids = self.s['ids']
        assert len(self.ids['train']) == 720 and len(self.ids['val']) == 96 and len(self.ids['test']) == 110

    def abspath(self, rel):
        return os.path.join(self.root, rel)

    def crops(self, vid, cam):
        return [self.abspath(p) for p in self.man[vid]['cam%d' % cam]]

    def middle(self, vid, cam):
        return self.crops(vid, cam)[2]

    # ---- training set: all train IDs, both cameras, all 5 crops ----
    def train_list(self):
        """[(path, pid_relabelled, camid0based, viewid)] ; camid 0=CAM1(rear) 1=CAM2(front)"""
        pid2lab = {v: i for i, v in enumerate(self.ids['train'])}
        out = []
        for v in self.ids['train']:
            for cam in (1, 2):
                for p in self.crops(v, cam):
                    out.append((p, pid2lab[v], cam - 1, 0))
        return out

    def eval_pairs(self, subset='test'):
        """query ids (CAM1) and the two gallery id lists."""
        q = list(self.ids[subset])
        g926 = sorted(self.man.keys(), key=int)
        g_small = list(self.ids[subset])
        return q, g926, g_small

    # ---- repo-internal (val) query/gallery lists, 1-crop ----
    def val_query_gallery(self):
        q = [(self.middle(v, 1), int(v), 0, 0) for v in self.ids['val']]
        g = [(self.middle(v, 2), int(v), 1, 0) for v in self.ids['val']]
        return q, g


def read_image(p):
    return Image.open(p).convert('RGB')


@torch.no_grad()
def embed_ids(model_fn, prot, ids, cam, batch=64, device='cuda'):
    """Return (feat_1crop [N,D], feat_5crop [N,D]) both L2-normalised.

    model_fn(list_of_PIL) -> torch.Tensor [B,D] (raw, un-normalised)
    camid passed implicitly by caller-side closure if needed.
    """
    paths, owner = [], []
    for i, v in enumerate(ids):
        for p in prot.crops(v, cam):
            paths.append(p); owner.append(i)
    feats = []
    for s in range(0, len(paths), batch):
        ims = [read_image(p) for p in paths[s:s + batch]]
        f = model_fn(ims)
        feats.append(f.float().cpu())
    F = torch.cat(feats).numpy()
    F = F.reshape(len(ids), 5, -1)
    f1 = F[:, 2, :]                          # middle crop
    f5 = F.mean(axis=1)                      # average of 5 embeddings
    return l2n(f1), l2n(f5)


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def rank_metrics(qf, qids, gf, gids):
    """qids/gids are lists of id strings. Exactly one match per query is asserted."""
    qf, gf = np.asarray(qf, np.float32), np.asarray(gf, np.float32)
    sim = qf @ gf.T
    order = np.argsort(-sim, axis=1, kind='stable')
    gidx = {v: i for i, v in enumerate(gids)}
    ranks = []
    for i, q in enumerate(qids):
        j = gidx[q]
        r = int(np.where(order[i] == j)[0][0]) + 1
        ranks.append(r)
    ranks = np.array(ranks)
    return dict(mAP=float((1.0 / ranks).mean()),
                R1=float((ranks <= 1).mean()),
                R5=float((ranks <= 5).mean()),
                R10=float((ranks <= 10).mean()),
                median_rank=float(np.median(ranks)))


def full_eval(model_fn1, model_fn2, prot, subset='test', batch=64, verbose=True):
    """model_fn1 embeds CAM1 images, model_fn2 embeds CAM2 images (they differ only if the
    model consumes a camera id). Returns nested dict [mode][gallery] -> metrics."""
    q, g926, gsm = prot.eval_pairs(subset)
    t0 = time.time()
    q1, q5 = embed_ids(model_fn1, prot, q, 1, batch)
    tq = time.time() - t0
    g1, g5 = embed_ids(model_fn2, prot, g926, 2, batch)
    small = np.array([g926.index(v) for v in gsm])
    res = {}
    for mode, (qf, gf) in [('1crop', (q1, g1)), ('5crop', (q5, g5))]:
        res[mode] = {
            'g926': rank_metrics(qf, q, gf, g926),
            'g%d' % len(gsm): rank_metrics(qf, q, gf[small], gsm),
        }
    res['_timing'] = {'query_embed_total_s': tq, 'n_query': len(q),
                      'per_query_5crop_s': tq / len(q), 'per_query_1crop_s': tq / len(q) / 5.0}
    if verbose:
        print(NOTE)
        for mode in ('1crop', '5crop'):
            for gk, m in res[mode].items():
                print(f"  {mode:6s} {gk:6s} mAP {m['mAP']*100:5.2f}  R1 {m['R1']*100:5.2f} "
                      f"R5 {m['R5']*100:5.2f} R10 {m['R10']*100:5.2f}")
    return res
