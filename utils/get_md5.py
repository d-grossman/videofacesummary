import hashlib

# Get MD5 hash of file contents
def file_digest(in_filename):
    BLOCKSIZE = 65536
    hasher = hashlib.md5()
    with open(in_filename, 'rb') as afile:
        buf = afile.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(BLOCKSIZE)
    return hasher.hexdigest()

# Get MD5 of file name
def pic2hash(v):
    v_str = '{0}'.format(v)
    v_str = v_str.encode("utf")
    return hashlib.md5(v_str).hexdigest()