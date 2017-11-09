import pickle
import sys
from collections import defaultdict

def reconsile(long_hash_f):
    o_data = defaultdict(dict)
    l_data = pickle.load(open(long_hash_f,'rb'))
    for entity_name in l_data:
        o_data[entity_name]['face_vec'] = l_data[entity_name]['face_vec']
        o_data[entity_name]['face_pic'] = l_data[entity_name]['face_pic']
        o_data[entity_name]['label'] = l_data[entity_name]['label']
        o_data[entity_name]['frame_pic'] = []
        o_data[entity_name]['videos'] = {}
        for vid in l_data[entity_name]['videos']:
            vid_file_hash = vid.split('_')[0]
            temp=list(l_data[entity_name]['videos'][vid])
            temp.sort(key=lambda tup: tup[0])
            o_data[entity_name]['videos'][vid_file_hash]= temp
    return o_data

if __name__ == "__main__":
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    out_data = reconsile(in_file)
    pickle.dump(out_data,open('out_file','wb'))
