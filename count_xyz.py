import sys
from ase.io import read, write
from collections import Counter as C

fname = sys.argv[1]
atoms = read(fname, index=':')
print(len(atoms), 'total configurations, of which:')

sizes_list = [ len(conf) for conf in atoms ]
count = C(sizes_list)

for size, times in sorted(count.items()):
  print(f"{times} times with {size} atoms")

if len(sys.argv) > 2 and sys.argv[2] == "split":
    bulk, surface, clusters = [], [], []

    for conf in atoms:
        n = len(conf)
        if n in (31, 32):
            bulk.append(conf)
        elif n in (63, 54, 80, 112):
            surface.append(conf)
        else:
            clusters.append(conf)

    if bulk:
        write("bulk.xyz", bulk)
        print(f"Wrote {len(bulk)} bulk configurations to bulk.xyz")
    if surface:
        write("surface.xyz", surface)
        print(f"Wrote {len(surface)} surface configurations to surface.xyz")
    if clusters:
        write("clusters.xyz", clusters)
        print(f"Wrote {len(clusters)} cluster configurations to clusters.xyz")
