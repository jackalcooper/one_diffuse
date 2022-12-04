import re

import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

if not os.isatty(sys.stdin.fileno()):
    txt = sys.stdin.readlines()
    txt = "".join(txt)
else:
    print("Skip, so it doesn't hang")
    exit(1)


# %[a-zA-Z_0-9#]+
# output_lbns = output_lbns
# scope_symbol_id = scope_symbol_id
# op_name = "OP_NAME"
txt = re.sub("%[a-zA-Z_0-9#]+", "%VALUE", txt)
txt = re.sub("output_lbns = \[.+?\]", "OUTPUT_LBNS", txt)
txt = re.sub("op_name = \".+?\"", "OP_NAME", txt)
txt = re.sub("scope_symbol_id = \d.+ : i64", "SCOPE_SYMBOL_ID", txt)
f = open(args.output, "w")
f.write(txt)
f.close()
