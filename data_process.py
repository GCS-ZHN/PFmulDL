import pandas as pd
import numpy as np

import pickle
def uni2pkl(uniprot_path, out_data_path):
    with open("data/go.pkl", 'rb') as file:
        go = pickle.loads(file.read())

    proteins, sequences, annotations, orgs = load_data(uniprot_path)
    df = pd.DataFrame({
        'proteins': proteins,
        'sequences': sequences,
        'annotations': annotations,
        'orgs': orgs
    })

    print('筛选包含试验注释的蛋白')
    index = []
    annotations = []
    for i, row in enumerate(df.itertuples()):
        annots = []
        for annot in row.annotations:
            go_id, code = annot.split('|')
            if is_exp_code(code):
                annots.append(go_id)
        if len(annots) == 0:
            continue
        index.append(i)
        annotations.append(annots)
    df = df.iloc[index]
    df = df.reset_index()
    df['exp_annotations'] = annotations

    print('prop annotations')
    prop_annotations = []
    for i, row in tqdm(df.iterrows()):
        # Propagate annotations
        annot_set = set()
        annots = row['exp_annotations']
        for go_id in annots:
            annot_set |= go.get_anchestors(go_id)
        annots = list(annot_set)
        prop_annotations.append(annots)
    df['prop_annotations'] = prop_annotations
    df.to_pickle(out_data_path)


def load_data(file):
    proteins = list()
    sequences = list()
    annotations = list()
    orgs = list()
    with gzip.open(file, 'rt') as f:
        prot_id = ''
        seq = ''
        org = ''
        annots = list()
        for line in tqdm(f):
            items = line.strip().split('   ')
            if items[0] == 'ID' and len(items) > 1:
                if prot_id != '':
                    proteins.append(prot_id)
                    sequences.append(seq)
                    annotations.append(annots)
                    orgs.append(org)
                prot_id = items[1]
                annots = list()
                seq = ''
            elif items[0] == 'OX' and len(items) > 1:
                if items[1].startswith('NCBI_TaxID='):
                    org = items[1][11:]
                    end = org.find(' ')
                    org = org[:end]
                else:
                    org = ''
            elif items[0] == 'DR' and len(items) > 1:
                items = items[1].split('; ')
                if items[0] == 'GO':
                    go_id = items[1]
                    code = items[3].split(':')[0]
                    annots.append(go_id + '|' + code)
            elif items[0] == 'SQ':
                seq = next(f).strip().replace(' ', '')
                while True:
                    sq = next(f).strip().replace(' ', '')
                    if sq == '//':
                        break
                    else:
                        seq += sq
        proteins.append(prot_id)
        sequences.append(seq)
        annotations.append(annots)
        orgs.append(org)
    return proteins, sequences, annotations, orgs