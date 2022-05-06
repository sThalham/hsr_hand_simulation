import os
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))
grasp_dir = os.path.join(base_dir, "grasp_poses")

def load_pickle_data(f_name):
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    return pickle_data


if __name__ == '__main__':
    frame_ids = sorted(os.listdir(os.path.join(grasp_dir)))    

    for f in frame_ids:
        print('File: %s' % f)

        f_name = os.path.join(grasp_dir, f)
        grasp = load_pickle_data(f_name)

        print(grasp)
