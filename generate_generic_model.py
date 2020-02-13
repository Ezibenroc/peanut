'''
This script is intented to generate a dgemm model based on given meta-paramters.
'''

import argparse
import sys
import datetime
import time
import yaml
import cashew
import numpy


def compute_reg(args):
    alpha = 2e-9/args.avg_gflops
    beta = args.avg_latency
    result = {
        'info': {
            'avg_gflops': args.avg_gflops,
            'avg_latency': args.avg_latency,
            'heterogeneity_coefficient': args.heterogeneity_coefficient,
            'variability_coefficient': args.variability_coefficient,
            'nb_nodes': args.nb_nodes,
        }
    }
    result['model'] = []
    alpha_p = numpy.random.normal(alpha, alpha*args.heterogeneity_coefficient, args.nb_nodes)
    beta_p  = numpy.random.normal(beta,  beta*args.heterogeneity_coefficient,  args.nb_nodes)
    for node in range(args.nb_nodes):
        reg = {}
        reg['intercept'] = beta_p[node]
        reg['mnk'] = alpha_p[node]
        reg['intercept_residual'] = beta_p[node] * args.variability_coefficient
        reg['mnk_residual'] = alpha_p[node] * args.variability_coefficient
        reg['avg_gflops'] = 2e-9/alpha_p[node]
        for key, val in reg.items():
            reg[key] = float(val)
        reg['node'] = node
        result['model'].append(reg)
    return result

def strict_positive_float(s):
    x = float(s)
    if x <= 0:
        raise ValueError()
    return x

def positive_float(s):
    x = float(s)
    if x < 0:
        raise ValueError()
    return x

def strict_positive_int(s):
    x = int(s)
    if x <= 0:
        raise ValueError()
    return x

def main():
    parser = argparse.ArgumentParser(description='Generic model generation for dgemm')
    parser.add_argument('avg_gflops', type=strict_positive_float,
                        help='Average dgemm performance (in Gflop/s) of the cluster.')
    parser.add_argument('avg_latency', type=positive_float,
                        help='Average dgemm latency (in s) of the cluster.')
    parser.add_argument('heterogeneity_coefficient', type=positive_float,
                        help='Coefficient of heterogeneity of the cluster.')
    parser.add_argument('variability_coefficient', type=positive_float,
                        help='Coefficient of variability of the cluster.')
    parser.add_argument('nb_nodes', type=strict_positive_int,
                        help='Number of nodes of the cluster.')
    parser.add_argument('output', type=argparse.FileType('w'))
    args = parser.parse_args()
    reg = compute_reg(args)
    reg['metadata'] = {
        'file_creation_date': str(datetime.datetime.now()),
        'granularity': 'node',
        'comment': 'generic model',
    }
    yaml.dump(reg, args.output)


if __name__ == '__main__':
    main()
