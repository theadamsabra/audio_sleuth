import os

def find_all_wav_files(dir_:str):
    '''Find all wav files in directory'''
    files = []
    
    # Walk across all dir/subdirs
    for root, _, filenames in os.walk(dir_):
        # Get the full paths of the wav if it is in this dir and keep it
        tmp = [os.path.join(root, filename) for filename in filenames if '.wav' in filename]
        files += tmp
    return files