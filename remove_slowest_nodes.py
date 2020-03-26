'''
This script takes a DGEMM model as a yaml file, remove the N slowest nodes and dump the result in another yaml file.
'''

import sys
import datetime
import time
import yaml


def read_model(filename):
    with open(filename) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def get_N_slowest(model, key='mnk', N=1):
    nodes = [(n['node'], n[key]) for n in model['model']]
    nodes.sort(key=lambda n: -n[1])
    return [n[0] for n in nodes[:N]]


def remove_nodes(model, nodes):
    nodes = set(nodes)
    rm_nodes = set()
    old_i = 0
    new_i = 0
    reg = model['model']
    while old_i < len(reg):
        if old_i not in nodes:
            reg[new_i] = reg[old_i]
            reg[new_i]['node'] = new_i
            reg[new_i]['old_node'] = old_i
            new_i += 1
        else:
            rm_nodes.add(old_i)
        old_i += 1
    assert rm_nodes == nodes
    model['model'] = reg[:new_i]
    model['info']['nb_nodes'] = new_i
    model['info']['nb_removed_nodes'] = len(nodes)
    model['info']['removed_nodes'] = list(sorted(nodes))


def main(input_file, output_file, nb_nodes):
    t1 = time.time()
    model = read_model(input_file)
    old_min_perf = min(n['avg_gflops'] for n in model['model'])
    nodes = get_N_slowest(model, N=nb_nodes)
    remove_nodes(model, nodes)
    new_min_perf = min(n['avg_gflops'] for n in model['model'])
    model['metadata']['original_file_creation_date'] = model['metadata']['file_creation_date']
    model['metadata']['file_creation_date'] = str(datetime.datetime.now())
    with open(output_file, 'w') as f:
        yaml.dump(model, f)
    t2 = time.time()
    print(f'Removed the slowest {nb_nodes} nodes in  {t2-t1:.2f} seconds')
    print(f'The minimum performance increased from {old_min_perf:.2f} Gflop/s to {new_min_perf:.2f} Gflop/s')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(f'Syntax: {sys.argv[0]} <nb_nodes> <input_file> <output_file>')
    nb_nodes = int(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    main(input_file, output_file, nb_nodes)
