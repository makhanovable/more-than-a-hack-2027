import os
import shutil
import subprocess
import urllib.request

from . import idsse


def _fetch(url, target):
    partial = target + ".part"
    if shutil.which("curl"):
        subprocess.run(["curl", "-sS", "-L", "--fail", "--retry", "3", "-C", "-", "-o", partial, url],
                       check=True)
    else:
        with urllib.request.urlopen(url) as response, open(partial, "wb") as handle:
            shutil.copyfileobj(response, handle, 1 << 20)
    os.replace(partial, target)


def fetch_all(data_dir=idsse.DATA_DIR, verbose=True):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    for match_id in idsse.MATCHES:
        for name in idsse.file_names(match_id):
            target = os.path.join(data_dir, name)
            if idsse._complete(target):
                if verbose:
                    print("  have     {}".format(name))
                continue
            if verbose:
                print("  fetching {} ...".format(name))
            _fetch(idsse.BASE_URL + name, target)
            if not idsse._complete(target):
                raise IOError("{} did not download completely".format(name))
    return idsse.available(data_dir)
