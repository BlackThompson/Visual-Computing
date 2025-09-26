#!/usr/bin/env python3
import os
import sys
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def read_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

def plot_kp_counts(detect_csv, out_dir):
    rows = read_csv(detect_csv)
    # aggregate by detector and image_role
    agg = defaultdict(list)
    for r in rows:
        if r.get('image_role') in ('pano','new'):
            agg[r['detector']].append(int(r['num_keypoints']))
    labels = sorted(agg.keys())
    vals = [np.mean(agg[k]) if agg[k] else 0 for k in labels]
    plt.figure(figsize=(6,4))
    plt.bar(labels, vals)
    plt.ylabel('Average keypoints per image')
    plt.title('Keypoints by detector')
    ensure_dir(out_dir)
    plt.savefig(os.path.join(out_dir, 'kp_counts_bar.png'), bbox_inches='tight')
    plt.close()

def plot_thresh_curves(ransac_csv, out_dir):
    rows = read_csv(ransac_csv)
    by_det = defaultdict(lambda: defaultdict(list))
    for r in rows:
        det = r['detector']
        th = float(r['thresh_px'])
        by_det[det]['th'].append(th)
        by_det[det]['inliers'].append(int(r['inliers']))
        by_det[det]['err'].append(float(r['avg_reproj_error_px']))
        by_det[det]['time'].append(float(r['ransac_time_ms']))
    ensure_dir(out_dir)
    for det, d in by_det.items():
        idx = np.argsort(d['th'])
        th = np.array(d['th'])[idx]
        for key, name in [('inliers','inliers'), ('err','avg_reproj_error_px'), ('time','ransac_time_ms')]:
            y = np.array(d[key])[idx]
            plt.figure(figsize=(6,4))
            plt.plot(th, y, marker='o')
            plt.xlabel('Threshold (px)')
            plt.ylabel(name)
            plt.title(f'{name} vs threshold ({det})')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(out_dir, f'{det}_{key}_vs_thresh.png'), bbox_inches='tight')
            plt.close()

def plot_blending_quality(stitch_csv, out_dir):
    rows = read_csv(stitch_csv)
    agg = defaultdict(lambda: defaultdict(list))  # det -> blend -> list
    for r in rows:
        agg[r['detector']][r['blending']].append(float(r['seam_error_mean']))
    labels = sorted(agg.keys())
    blends = sorted({b for d in agg.values() for b in d.keys()})
    x = np.arange(len(labels))
    width = 0.35 if len(blends)==2 else 0.8/len(blends)
    ensure_dir(out_dir)
    plt.figure(figsize=(7,4))
    for i, b in enumerate(blends):
        vals = [np.mean(agg[det][b]) if agg[det][b] else 0 for det in labels]
        plt.bar(x + i*width, vals, width=width, label=b)
    plt.xticks(x + width*(len(blends)-1)/2, labels)
    plt.ylabel('Seam error mean')
    plt.title('Blending quality by detector')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'blending_quality_bar.png'), bbox_inches='tight')
    plt.close()

def main():
    if len(sys.argv) < 2:
        print('Usage: plot_results.py <run_dir> [out_dir]')
        return
    run_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run_dir, 'report_figs')
    detect_csv = os.path.join(run_dir, 'detect_describe.csv')
    ransac_csv = os.path.join(run_dir, 'ransac.csv')
    stitch_csv = os.path.join(run_dir, 'stitch.csv')

    plot_kp_counts(detect_csv, out_dir)
    plot_thresh_curves(ransac_csv, out_dir)
    plot_blending_quality(stitch_csv, out_dir)
    print('Saved figures to', out_dir)

if __name__ == '__main__':
    main()


