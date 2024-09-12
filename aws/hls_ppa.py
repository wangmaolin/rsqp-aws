import pandas as pd

def contains_partial_match(SearchNames, search_string):
    for item in SearchNames:
        if item in search_string and\
            bool(search_string) and\
            'Pipeline' not in search_string:
            return True
    return False

def extract_module_info(file_content, SearchNames):
    lines = file_content.split('\n')
    ReportDict = {'module_name':1,
        'iter_latency':6,
        'iter_interval':7,
        'bram':10,
        'dsp':11,
        'ff':12,
        'lut':13,
        'uram':14,
    }
    PPAdf = pd.DataFrame(columns=ReportDict.keys())

    for line in lines:
        lineSplit = line.split('|')       
        if len(lineSplit) < 2:
            continue
        if not contains_partial_match(SearchNames, lineSplit[1].strip()):
            continue
        insertDict= {key: lineSplit[value].strip() for key, value in ReportDict.items()}
        PPAdf.loc[len(PPAdf)]= insertDict

    return PPAdf

def main():
    rpt_path='./temp/csynth.rpt'

    # Read the content of the file
    with open(rpt_path, 'r') as file:
        file_content = file.read()

    SearchNames = ['cu_top',
                   'instruction_omega_fu',
                    'omega_compute_loop', 
                    'omega_ew_prod_loop', 
                    'omega_read_loop', 
                    'instruction_norm_inf_binary_tree',
                    'instruction_axpby_fu',
                    'axpby_loop',
                    'instruction_dot_fu',
                    'inst_dot_loop',
                    'instruction_vec_max_fu',
                    'vec_max_loop',
                    'instruction_vec_min_fu',
                    'vec_min_loop',
                    'instruction_ew_div_fu',
                    'ew_div_loop',
                    'instruction_ew_prod_fu',
                    'ew_prod_loop',
                    'instruction_cond_set_fu',
                    'cond_set_loop',
                    'instruction_scaled_add_fu',
                    'scaled_add_loop',
                   ]

    df = extract_module_info(file_content, SearchNames)
    df.to_csv('./temp/ppa.csv')

if __name__ == '__main__':
    main()
