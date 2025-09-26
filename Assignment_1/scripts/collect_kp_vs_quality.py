#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import csv
import glob

BIN = "/mnt/d/Code/Course/Visual Computing/Assignment_1/build/panorama"
OUT_ROOT = "/mnt/d/Code/Course/Visual Computing/Assignment_1/results"
FINAL_DIR = "/mnt/d/Code/Course/Visual Computing/Assignment_1/final_results"

os.makedirs(FINAL_DIR, exist_ok=True)

HEADER = [
    'run_dir','detector','ratio','thresh_px',
    'kp1','kp2','matches_raw','matches_ratio','inliers','inlier_ratio','median_desc_dist','match_time_ms'
]


def latest_run_dir(root):
    runs = [d for d in glob.glob(os.path.join(root, 'run_*')) if os.path.isdir(d)]
    if not runs:
        return None
    runs.sort(key=os.path.getmtime)
    return runs[-1]


def read_csv_one(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def median(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return 0.5 * (float(s[mid-1]) + float(s[mid]))


def collect_from_run(run_dir):
    params = {}
    # detector, ratio, thresh
    p = read_csv_one(os.path.join(run_dir, 'matching.csv'))
    r = read_csv_one(os.path.join(run_dir, 'ransac.csv'))
    d = read_csv_one(os.path.join(run_dir, 'detect_describe.csv'))

    if not p or not r or not d:
        return None

    detector = p[-1]['detector']
    ratio = float(p[-1]['ratio'])
    thresh = float(r[-1]['thresh_px'])

    # kp1/kp2: pano/new 两行在 detect_describe.csv
    kp1 = 0; kp2 = 0
    for row in d:
        if row['image_role'] == 'pano':
            kp1 = int(row['num_keypoints'])
        elif row['image_role'] == 'new':
            kp2 = int(row['num_keypoints'])

    matches_raw = int(p[-1]['raw_matches'])
    matches_ratio = int(p[-1]['kept_matches'])
    raw_match_time_ms = float(p[-1]['raw_match_time_ms']) if 'raw_match_time_ms' in p[-1] else 0.0
    filter_time_ms = float(p[-1]['filter_time_ms']) if 'filter_time_ms' in p[-1] else 0.0
    match_time_ms = raw_match_time_ms + filter_time_ms

    inliers = int(float(r[-1]['inliers']))
    inlier_ratio = float(r[-1]['inlier_ratio']) if matches_ratio > 0 else 0.0

    # median_desc_dist from kept_distances.csv
    kept_d_path = os.path.join(run_dir, 'kept_distances.csv')
    kept_d = []
    if os.path.exists(kept_d_path):
        with open(kept_d_path) as f:
            for line in f:
                line=line.strip()
                if line:
                    kept_d.append(float(line))
    med_dist = median(kept_d)

    return {
        'run_dir': os.path.basename(run_dir),
        'detector': detector,
        'ratio': ratio,
        'thresh_px': thresh,
        'kp1': kp1,
        'kp2': kp2,
        'matches_raw': matches_raw,
        'matches_ratio': matches_ratio,
        'inliers': inliers,
        'inlier_ratio': inlier_ratio,
        'median_desc_dist': med_dist,
        'match_time_ms': match_time_ms,
    }


def upgrade_existing_rows(rows):
    # Ensure each row has all HEADER fields; compute missing match_time_ms
    upgraded = []
    for r in rows:
        rd = dict(r)
        if 'match_time_ms' not in rd or rd['match_time_ms'] == '':
            run_dir = rd.get('run_dir','')
            det = rd.get('detector','')
            mrows = read_csv_one(os.path.join(OUT_ROOT, run_dir, 'matching.csv'))
            if mrows:
                raw_t = float(mrows[-1].get('raw_match_time_ms','0') or 0)
                fil_t = float(mrows[-1].get('filter_time_ms','0') or 0)
                rd['match_time_ms'] = raw_t + fil_t
            else:
                rd['match_time_ms'] = ''
        # ensure required fields exist
        for h in HEADER:
            if h not in rd:
                rd[h] = ''
        upgraded.append(rd)
    return upgraded

def append_summary(row):
    out_csv = os.path.join(FINAL_DIR, 'kp_vs_quality.csv')
    if os.path.exists(out_csv):
        # read all, upgrade header/rows if needed, replace or append by run_dir
        with open(out_csv, newline='') as f:
            reader = csv.DictReader(f)
            existing = list(reader)
        # upgrade header fields in memory
        existing = upgrade_existing_rows(existing)
        # replace row with same run_dir if exists
        key = row['run_dir']
        replaced = False
        for i, r in enumerate(existing):
            if r.get('run_dir') == key:
                existing[i] = row
                replaced = True
                break
        if not replaced:
            existing.append(row)
        # write back with new HEADER
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            for r in existing:
                # ensure all fields present and strip extras
                clean = {h: r.get(h, '') for h in HEADER}
                w.writerow(clean)
        print('Updated:', out_csv)
    else:
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            w.writeheader()
            # strip extras from new row just in case
            clean = {h: row.get(h, '') for h in HEADER}
            w.writerow(clean)
        print('Created:', out_csv)

def plot_run_histograms(run_dir, detector):
    import matplotlib.pyplot as plt
    hist_dir = os.path.join(FINAL_DIR, 'histograms')
    os.makedirs(hist_dir, exist_ok=True)
    def plot_one(csv_path, title, out_name, bins=50):
        vals = []
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                for line in f:
                    s = line.strip()
                    if s:
                        vals.append(float(s))
        if not vals:
            return
        plt.figure(figsize=(6,4))
        plt.hist(vals, bins=bins, color='#1f77b4', alpha=0.85)
        plt.title(f'{title} ({detector})')
        plt.xlabel('distance')
        plt.ylabel('count')
        out_path = os.path.join(hist_dir, out_name)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
    run_id = os.path.basename(run_dir)
    plot_one(os.path.join(run_dir,'raw_distances.csv'), 'Raw match distances', f'{run_id}_{detector}_raw_hist.png')
    plot_one(os.path.join(run_dir,'kept_distances.csv'), 'Kept match distances', f'{run_id}_{detector}_kept_hist.png')


def run_once(img1, img2, det='sift', ratio='0.8', th='4', blend='feather', set_id='exp', pair_id='img01_to_img02'):
    cmd = [BIN,
           '--set', set_id, '--pair', pair_id,
           '--det', det, '--blend', blend,
           '--ratio', str(ratio), '--ransac', '2000', '--th', str(th), '--debug',
           img1, img2]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    run_dir = latest_run_dir(OUT_ROOT)
    return run_dir


def main():
    if len(sys.argv) < 3:
        print('Usage: collect_kp_vs_quality.py <img1> <img2> [det sift|orb|akaze] [ratio 0.8] [th 4]')
        return
    img1, img2 = sys.argv[1], sys.argv[2]
    det = sys.argv[3] if len(sys.argv) > 3 else 'sift'
    ratio = sys.argv[4] if len(sys.argv) > 4 else '0.8'
    th = sys.argv[5] if len(sys.argv) > 5 else '4'

    run_dir = run_once(img1, img2, det=det, ratio=ratio, th=th)
    row = collect_from_run(run_dir)
    if row is None:
        print('Failed to collect metrics from', run_dir)
        return
    append_summary(row)
    print('Done. Run dir =', run_dir)

if __name__ == '__main__':
    main()
