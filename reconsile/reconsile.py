import pickle
import sys
from collections import defaultdict

def reconsile(long_hash_f):
    o_data = defaultdict(dict)
    l_data = pickle.load(open(long_hash_f,'rb'))
    for key in l_data:
        new_key = key.split('_')[0]
        o_data[new_key]['face_vec'] = l_data[key]['face_vec']
        o_data[new_key]['face_pic'] = l_data[key]['face_pic']
        o_data[new_key]['label'] = l_data[key]['label']
        o_data[new_key]['frame_pic'] = []
        o_data[new_key]['videos'] = {}
        for vid in l_data[key]['videos']:
            vid_key = vid.split('_')[0]
            temp=list(l_data[key]['videos'][vid])
            temp.sort(key=lambda tup: tup[0])
            o_data[new_key]['videos'][vid_key]= temp
    return o_data

if __name__ == "__main__":
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    out_data = reconsile(in_file)
    pickle.dump(out_data,open('out_file','wb'))
