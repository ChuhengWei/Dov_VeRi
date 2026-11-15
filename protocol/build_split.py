"""Build the SESSION-disjoint split and the image manifest. Read-only w.r.t. CAM1/CAM2."""
import os, json, glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CAM = {1: os.path.join(ROOT, 'CAM1'), 2: os.path.join(ROOT, 'CAM2')}

TRAIN_SESS = [1, 2, 3, 5, 7, 8, 10, 12]
VAL_SESS   = [6, 9, 11]
TEST_SESS  = [4, 13]


def frame_idx(p):
    return int(os.path.splitext(os.path.basename(p))[0])


def crops(cam, vid):
    fs = sorted(glob.glob(os.path.join(CAM[cam], vid, '*.jpg')), key=frame_idx)
    return fs


def main():
    ids1 = {d for d in os.listdir(CAM[1]) if os.path.isdir(os.path.join(CAM[1], d))}
    ids2 = {d for d in os.listdir(CAM[2]) if os.path.isdir(os.path.join(CAM[2], d))}
    common = sorted(ids1 & ids2, key=int)

    dropped = []
    usable = []
    for v in common:
        n1, n2 = len(crops(1, v)), len(crops(2, v))
        if n1 == 5 and n2 == 5:
            usable.append(v)
        else:
            dropped.append({'id': v, 'n_cam1': n1, 'n_cam2': n2})
    print(f'ids in both cams: {len(common)}; usable: {len(usable)}; dropped: {dropped}')
    assert len(usable) == 926, len(usable)

    sess = {v: int(v[:-3]) for v in usable}
    split = {}
    for name, ss in [('train', TRAIN_SESS), ('val', VAL_SESS), ('test', TEST_SESS)]:
        split[name] = sorted([v for v in usable if sess[v] in ss], key=int)

    st, sv, se = map(set, (split['train'], split['val'], split['test']))
    assert st & sv == set() and st & se == set() and sv & se == set(), 'splits overlap'
    assert st | sv | se == set(usable), 'splits do not cover all usable ids'
    # session disjointness
    assert set(TRAIN_SESS) & set(VAL_SESS) == set() and set(TRAIN_SESS) & set(TEST_SESS) == set() \
        and set(VAL_SESS) & set(TEST_SESS) == set()
    print({k: len(v) for k, v in split.items()})
    assert (len(split['train']), len(split['val']), len(split['test'])) == (720, 96, 110)

    manifest = {v: {'session': sess[v],
                    'cam1': [os.path.relpath(p, ROOT) for p in crops(1, v)],
                    'cam2': [os.path.relpath(p, ROOT) for p in crops(2, v)]}
                for v in usable}

    out = {'root': ROOT,
           'sessions': {'train': TRAIN_SESS, 'val': VAL_SESS, 'test': TEST_SESS},
           'ids': split,
           'dropped': dropped,
           'n_ids': {k: len(v) for k, v in split.items()},
           'manifest': manifest}
    with open(os.path.join(os.path.dirname(__file__), '..', 'split.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote split.json')

    # sanity: middle crop choice
    v = split['test'][0]
    print('example', v, 'cam1 crops', [os.path.basename(p) for p in manifest[v]['cam1']],
          '-> middle', os.path.basename(manifest[v]['cam1'][2]))


if __name__ == '__main__':
    main()
