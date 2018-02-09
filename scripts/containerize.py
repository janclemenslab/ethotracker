#!/usr/bin/env python3
import sys
inp = sys.argv[1]
out = sys.argv[2]

cmd = 'ffmpeg -i {inp} -vcodec copy {out}'.format(inp=inp, out=out)
print(cmd)
os.system(cmd)
print('done')