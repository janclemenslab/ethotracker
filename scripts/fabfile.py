from fabric.api import *

REMOTE = 'clemens10@gwdu102.gwdg.de:/scratch/clemens10/'
LOCAL = '/Volumes/ukme04/#Common/'
FOLDER = 'playback'
# 


def _rsync(source, target, delete=False, excludes=[]):
    RSYNC = 'rsync -avHz --update --progress'
   # delete files on target
    if delete:
        RSYNC += ' --delete'

    # exclude files/directories
    for exclude in excludes:
        RSYNC += ' --exclude="{0}"'.format(exclude)

    RSYNC += " {0} {1}".format(source + FOLDER, target)
    local(RSYNC)

def push(delete=False, excludes=[".*", ".*/"]):
    _rsync(LOCAL, REMOTE, delete, excludes)


def pull(delete=False, excludes=[".*", ".*/"]):
    _rsync(REMOTE, LOCAL, delete, excludes)
