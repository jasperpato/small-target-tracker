import sys
from scipy.stats import norm


def change_thresholds(range=0.9):
  '''
  Use current cue results stores in cue_results.txt to update the thresholds based on range parameter.
  '''
  
  with open('results/cue_results.txt', 'r') as f:
    area_avg, area_std, ext_avg, ext_std, alen_avg, alen_std, ecc_avg, ecc_std = [float(n) for n in f.read().split(',')[2:]]

  t1_area = norm.ppf((1-range)/2, loc=area_avg, scale=area_std)
  t2_area = 2 * area_avg - t1_area
  t1_ext = norm.ppf((1-range)/2, loc=ext_avg, scale=ext_std)
  t2_ext = 2 * ext_avg - t1_ext
  t1_alen = norm.ppf((1-range)/2, loc=alen_avg, scale=alen_std)
  t2_alen = 2 * alen_avg - t1_alen
  t1_ecc = norm.ppf((1-range)/2, loc=ecc_avg, scale=ecc_std)
  t2_ecc = 2 * ecc_avg - t1_ecc

  with open('results/cue_thresholds.txt', 'w') as f:
    f.write(f'{t1_area}, {t2_area}, {t1_ext}, {t2_ext}, {t1_alen}, {t2_alen}, {t1_ecc}, {t2_ecc}')


if __name__ == '__main__':
    change_thresholds(float(sys.argv[1]))