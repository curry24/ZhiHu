import os
def ensure_path(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def filter_doc(document):
    words = document.strip().decode('utf8').split()
    words = [w for w in words if len(w) > 0 and (not w.startswith('http'))]
    return words