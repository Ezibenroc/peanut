import sys
import yaml
from peanut.smpi_hpl import model_to_c_code


def main(model_file, code_file):
    with open(model_file) as f:
        model = yaml.load(f, Loader=yaml.SafeLoader)
    code = model_to_c_code(model['model'])
    with open(code_file, 'w') as f:
        f.write(code)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Syntax: %s <model_file> <code_file>' % sys.argv[0])
    model_file = sys.argv[1]
    code_file = sys.argv[2]
    if not model_file.endswith('.yaml'):
        sys.exit('File %s must be a .yaml file' % model_file)
    if not code_file.endswith('.c'):
        sys.exit('File %s must be a .c file' % code_file)
    main(model_file, code_file)
