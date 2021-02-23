import os
import pickle

base_dir = '/home/kw/0_code/grasping/hsr_hand_simulation/hsr_hand_simulation/grasp_poses/'

def load_pickle_data(f_name):
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)
    return pickle_data


if __name__ == '__main__':
    frame_ids = sorted(os.listdir(os.path.join(base_dir)))    

    for f in frame_ids:
        print('File: %s' % f)

        f_name = os.path.join(base_dir, f)
        grasp = load_pickle_data(f_name)

        print(grasp)
