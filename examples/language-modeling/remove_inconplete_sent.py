import spacy
import json
import sys
from tqdm import tqdm

nlp = spacy.load("en_core_web_lg")

f = open('VisualNews_UMD_gt_partition_pp_fix_gpt2.json')
dataset_all_gpt = json.loads(f.read())
print(f'number of datapoints before {len(dataset_all_gpt)}')

lens = []
dataset_all_gpt_fix = []
for elt in tqdm(dataset_all_gpt):
    assert 'gpt2_output' in elt, elt['id']
    lens.append(len(elt['gpt2_output'].split()))
    doc = nlp(elt['gpt2_output'])
    sents = [sent.text for sent in doc.sents]
    if sents[-1][-1] not in {'.' '!', '?'}:
        # print(sents[-1])
        sents = sents[:-1]
        elt['gpt2_output'] = ' '.join(sents)
        assert lens[-1] != len(elt['gpt2_output']), 'no len change'
        # print(elt['gpt2_output'])
        dataset_all_gpt_fix.append(elt)
f.close()
f = open('VisualNews_UMD_gt_partition_pp_fix_gpt2_sents.json', 'w')
f.write(json.dumps(dataset_all_gpt_fix, indent=4))





