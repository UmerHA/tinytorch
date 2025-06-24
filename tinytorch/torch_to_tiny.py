import re
from collections import namedtuple

Rule = namedtuple('Rule', ['pattern', 'replacement'])

def make_torch_tiny(input_filename, output_filename):
    with open(input_filename, 'r') as f: code = f.read()
    # Define transformation rules as a list of Rule namedtuples
    transformation_rules = [
        Rule('#include <torch/extension.h>', '#include "tinytorch.h"'),
        Rule(r'#include\s*<torch/.*>', ''),
        Rule('TORCH_CHECK', 'TINYTORCH_CHECK'),
        Rule(r'#define\s+AT_DISPATCH_CASE.*\n', ''),
        Rule(r'#define\s+AT_DISPATCH_SWITCH.*\n', ''),
        Rule(r'#define\s+AT_DISPATCHER_CASE.*\n', ''),
        Rule(r'#define\s+AT_DISPATCHER.*\n', ''),
        Rule(r'AT_DISPATCHER\((.*?)\.scalar_type\(\)', r'TINYTORCH_DISPATCH_BY_TYPE(\1.dtype()'),
        Rule(r'\btorch::Tensor\b', 'TinyTensor'),
        Rule('torch::zeros', 'zeros'),
        Rule(r'PYBIND11_MODULE\s*\([\s\S]*?\)\s*\{[\s\S]*?\}', ''),
        Rule(r'TORCH_LIBRARY\s*\([\s\S]*?\)\s*\{[\s\S]*?\}', ''),
        Rule(r'TORCH_LIBRARY_IMPL\s*\([\s\S]*?\)\s*\{[\s\S]*?\}', ''),
        Rule(r'.sizes\(\)', r'.sizes_as_str()')
    ]
    for rule in transformation_rules: code = re.sub(rule.pattern, rule.replacement, code)
    with open(output_filename, 'w') as f: f.write(code)

make_torch_tiny('<your_file>.cu', '<your_file>_tiny.cu')
