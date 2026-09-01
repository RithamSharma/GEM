import sys
import re
import json
import os

def sanitize_gv(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    out = []
    i = 0
    in_param = False
    paren_depth = 0
    param_buffer = ""
    params_dict = {}

    while i < len(content):
        if not in_param:
            if content[i:i+2] == "#(":
                in_param = True
                paren_depth = 1
                i += 2
                param_buffer = "#("
            else:
                out.append(content[i])
                i += 1
        else:
            param_buffer += content[i]
            if content[i] == "(":
                paren_depth += 1
            elif content[i] == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    in_param = False
                    
                    # Find the instance name that follows this parameter block
                    j = i + 1
                    while j < len(content) and content[j].isspace():
                        j += 1
                    
                    inst_name_start = j
                    while j < len(content) and not content[j].isspace() and content[j] != "(":
                        j += 1
                    inst_name = content[inst_name_start:j]
                    
                    # Parse the parameters from the buffer: .KEY(VALUE)
                    parsed_params = {}
                    for match in re.finditer(r'\.\s*([A-Za-z0-9_]+)\s*\(\s*(.*?)\s*\)', param_buffer):
                        key = match.group(1)
                        val = match.group(2)
                        # Clean up Verilog string literals if present
                        if val.startswith('"') and val.endswith('"'):
                            val = val[1:-1]
                        parsed_params[key] = val
                        
                    params_dict[inst_name] = parsed_params
            i += 1

    json_path = os.path.join(os.path.dirname(filepath), "params.json")
    with open(json_path, "w") as f:
        json.dump(params_dict, f, indent=2)

    with open(filepath, "w") as f:
        f.write("".join(out))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 sanitize_gv.py <file.gv>")
        sys.exit(1)
    sanitize_gv(sys.argv[1])
