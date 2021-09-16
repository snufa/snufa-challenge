import os
import sys
import time
import urllib.request
import gzip, shutil
import hashlib
import argparse

from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import urlretrieve


# The functions used in this file to download the dataset are based on
# code from the keras library. Specifically, from the following file:
# https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/utils/data_utils.py

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    if count % 1000 == 0:
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                         (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(file_hash) == 64):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def get_file(fname,
             origin,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.data-cache')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.data-cache')
    datadir = os.path.join(datadir_base, cache_subdir)
    os.makedirs(datadir, exist_ok=True)

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated because the ' + hash_algorithm +
                      ' file hash does not match the original value of ' + file_hash +
                      ' so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
        try:
            urlretrieve(origin, fpath, reporthook)
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.msg))
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(fpath):
            os.remove(fpath)

    return fpath


def get_and_gunzip(origin, filename, md5hash=None, cache_dir=None, cache_subdir=None):
    gz_file_path = get_file(filename, origin, md5_hash=md5hash, cache_dir=cache_dir, cache_subdir=cache_subdir)
    hdf5_file_path = gz_file_path.rsplit(".", 1)[0]
    if not os.path.isfile(os.path.join(origin, hdf5_file_path)) or os.path.getctime(gz_file_path) > os.path.getctime(
            hdf5_file_path):
        print("\nDecompressing %s" % gz_file_path)
        with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
            # print(f_in, f_out)
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path


def get_dataset(cache_dir, cache_subdir, dataset="snufa100"):
    # The remote directory with the data files
    base_url = "https://compneuro.net/datasets/snufa100"

    # Retrieve MD5 hashes from remote
    response = urllib.request.urlopen("%s/md5sums.txt" % base_url)
    data = response.read()
    lines = data.decode('utf-8').split("\n")
    file_hashes = {line.split()[1]: line.split()[0] for line in lines if len(line.split()) == 2}
    # Download the dataset
    dataset = dataset.lower()
    if dataset == "snufa100":
        files = ["snufa100_train.h5.gz", ]
    elif dataset == "snufa100_sentences":
        files = ["snufa100_sentences_train.h5.gz", ]
    else:
        raise Exception("Incorrect Dataset, choose snufa100 or snufa100_sentences")
    fpaths = []
    for fn in files:
        origin = "%s/%s" % (base_url, fn)
        if os.path.exists(os.path.join(cache_dir, cache_subdir, fn)):
            print("File %s already exists, skipping" % (fn))
            fpaths.append(os.path.abspath(os.path.join(cache_dir, cache_subdir, fn)[:-3]))
        else:
            hdf5_file_path = get_and_gunzip(origin, fn, md5hash=file_hashes[fn], cache_dir=cache_dir,
                                            cache_subdir=cache_subdir)
            print("File %s decompressed to:" % (fn))
            print(hdf5_file_path)
            fpaths.append(os.path.abspath(hdf5_file_path))


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument('-dataset', help="SNUFA100 or SNUFA100_sentences", default="snufa100")
    parser.add_argument('-ddir', '--datadir', help="directory to store data", type=dir_path, default="./data")
    args = vars(parser.parse_args())

    cache_dir = args['datadir']
    cache_subdir = args['dataset']

    get_dataset(cache_dir, cache_subdir, dataset=args["dataset"])


if __name__ == "__main__":
    main()
