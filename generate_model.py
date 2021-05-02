'''
This script is intented to generate a dgemm model from a BLAS calibration archive.
'''

import sys
import datetime
import time
import yaml
import cashew
import numpy
from cashew import linear_regression as lr
from cashew import archive_extraction as ae

def my_dgemm_reg(df):
    df = df.copy()
    lr.compute_variable_products(df, 'mnk')
    reg = lr.compute_full_reg(df, 'duration', ['mnk', 'mn', 'mk', 'nk'])
    total_flop = (2 * df['m'] * df['n'] * df['k']).sum()
    total_time = df['duration'].sum()
    reg['mean_gflops'] = total_flop / total_time * 1e-9
    return reg

def compute_reg(df):
    reg = lr.regression(df, my_dgemm_reg)
    for tmp in reg:
        for key, val in tmp.items():
            if isinstance(val, (numpy.int, numpy.int64)):
                tmp[key] = int(tmp[key])
            elif isinstance(val, (numpy.float, numpy.float64)):
                tmp[key] = float(tmp[key])
    result = {'info': {}}
    for key in ['cluster', 'jobid', 'expfile_hash', 'start_time']:
        values = {tmp[key] for tmp in reg}
        assert len(values) == 1
        result['info'][key] = values.pop()
        for tmp in reg:
            del tmp[key]
    result['info']['experiment_date'] = str(datetime.datetime.fromtimestamp(result['info']['start_time']))
    del result['info']['start_time']
    avg_alpha = numpy.mean([tmp['mnk'] for tmp in reg])
    avg_beta  = numpy.mean([tmp['intercept'] for tmp in reg])
    var_coeff = numpy.mean([tmp['mnk_residual']/tmp['mnk'] for tmp in reg])
    het_coeff = numpy.std([tmp['mnk'] for tmp in reg]) / avg_alpha
    result['info'].update({
        'avg_gflops': float(2e-9/avg_alpha),
        'avg_latency': float(avg_beta),
        'heterogeneity_coefficient': float(het_coeff),
        'variability_coefficient': float(var_coeff),
        'nb_nodes': len(df['node'].unique()),
    })
    for tmp in reg:
        tmp['cpu_id'] = 2*tmp['node'] + tmp['cpu']  # see the function get_cpuid() in HPL_dgemm.c
    result['model'] = reg
    return result


def main(archive_file, model_file):
    t1 = time.time()
    df = ae.read_performance(archive_file)
    t2 = time.time()
    print('Extracted archive in %.2f seconds' % (t2-t1))
    reg = compute_reg(df)
    t3 = time.time()
    print('Computed model in %.2f seconds' % (t3-t2))
    reg['metadata'] = {
        'file_creation_date': str(datetime.datetime.now()),
        'archive_file': archive_file,
        'cashew_git': cashew.__git_version__,
        'granularity': 'cpu',
    }
    with open(model_file, 'w') as f:
        yaml.dump(reg, f)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Syntax: %s <archive_file> <model_file>' % sys.argv[0])
    archive_file = sys.argv[1]
    model_file = sys.argv[2]
    if not archive_file.endswith('.zip'):
        sys.exit('File %s must be a .zip file' % archive_file)
    if not model_file.endswith('.yaml'):
        sys.exit('File %s must be a .yaml file' % model_file)
    main(archive_file, model_file)
