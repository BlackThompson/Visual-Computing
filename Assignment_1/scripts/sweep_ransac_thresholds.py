#!/usr/bin/env python3
import os
import sys
import subprocess
import csv
import glob

BIN = "/mnt/d/Code/Course/Visual Computing/Assignment_1/build/panorama"
OUT_ROOT = "/mnt/d/Code/Course/Visual Computing/Assignment_1/results"
FINAL_DIR = "/mnt/d/Code/Course/Visual Computing/Assignment_1/final_results"

HEADER = [
    'run_dir','detector','thresh_px','ransac_iters',
    'inliers','inlier_ratio','avg_reproj_error_px','ransac_time_ms',
    'warp_time_ms','blend_time_ms','seam_error_mean','seam_error_max','out_w','out_h'
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

def append_row(row, out_csv):
    exists = os.path.exists(out_csv)
    with open(out_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if not exists:
            w.writeheader()
        w.writerow(row)

def run_once(img1, img2, th, iters=2000):
    cmd = [BIN,
           '--set', 'outroom', '--pair', 'img01_to_img02',
           '--det', 'sift', '--blend', 'feather', '--ratio', '0.8',
           '--ransac', str(iters), '--th', str(th), '--debug',
           img1, img2]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return latest_run_dir(OUT_ROOT)

def main():
    if len(sys.argv) < 3:
        print('Usage: sweep_ransac_thresholds.py <outroom_img1> <outroom_img2> [th1 th2 th3 th4 th5 ...]')
        return
    img1, img2 = sys.argv[1], sys.argv[2]
    ths = [float(t) for t in (sys.argv[3:] if len(sys.argv) > 3 else ['1','2','3','5','10'])]
    out_csv = os.path.join(FINAL_DIR, 'ransac_sweep_outroom_sift.csv')
    os.makedirs(FINAL_DIR, exist_ok=True)

    for th in ths:
        run_dir = run_once(img1, img2, th)
        ransac_rows = read_csv_one(os.path.join(run_dir, 'ransac.csv'))
        stitch_rows = read_csv_one(os.path.join(run_dir, 'stitch.csv'))
        match_rows = read_csv_one(os.path.join(run_dir, 'matching.csv'))
        if not ransac_rows or not stitch_rows or not match_rows:
            print('Missing CSVs in', run_dir)
            continue
        r = ransac_rows[-1]
        s = stitch_rows[-1]
        m = match_rows[-1]
        row = {
            'run_dir': os.path.basename(run_dir),
            'detector': r.get('detector','sift'),
            'thresh_px': r.get('thresh_px',''),
            'ransac_iters': r.get('iters',''),
            'inliers': r.get('inliers',''),
            'inlier_ratio': r.get('inlier_ratio',''),
            'avg_reproj_error_px': r.get('avg_reproj_error_px',''),
            'ransac_time_ms': r.get('ransac_time_ms',''),
            'warp_time_ms': s.get('warp_time_ms',''),
            'blend_time_ms': s.get('blend_time_ms',''),
            'seam_error_mean': s.get('seam_error_mean',''),
            'seam_error_max': s.get('seam_error_max',''),
            'out_w': s.get('out_w',''),
            'out_h': s.get('out_h',''),
        }
        append_row(row, out_csv)
        print('Appended:', out_csv)

if __name__ == '__main__':
    main()


