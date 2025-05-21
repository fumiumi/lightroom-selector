# Lightroomでは外部xmpは使えないので、これは使わない


import os
import shutil
import argparse

#!/usr/bin/env python3

def collect_xmp_files(xmp_dir):
  xmp_map = {}
  for root, _, files in os.walk(xmp_dir):
    for fname in files:
      if fname.lower().endswith('.xmp'):
        base = os.path.splitext(fname)[0]
        xmp_map[base] = os.path.join(root, fname)
  return xmp_map

def copy_xmp_to_dng_dirs(xmp_map, lrdata_dir):
  for root, _, files in os.walk(lrdata_dir):
    for fname in files:
      if fname.lower().endswith('.dng'):
        base = os.path.splitext(fname)[0]
        if base in xmp_map:
          src = xmp_map[base]
          dst = os.path.join(root, base + '.xmp')
          shutil.copy2(src, dst)
          print(f"Copied {src} -> {dst}")

def main():
  parser = argparse.ArgumentParser(
    description="Copy XMP files from xmp_out to matching DNG directories in a Lightroom Smart Previews.lrdata.")
  parser.add_argument(
    "xmp_out",
    help="Path to the directory containing .xmp files (e.g. ./xmp_out)")
  parser.add_argument(
    "lrdata",
    help="Path to the Lightroom Catalog Smart Previews.lrdata directory")
  args = parser.parse_args()

  if not os.path.isdir(args.xmp_out):
    parser.error(f"xmp_out directory not found: {args.xmp_out}")
  if not os.path.isdir(args.lrdata):
    parser.error(f"LRData directory not found: {args.lrdata}")

  xmp_map = collect_xmp_files(args.xmp_out)
  if not xmp_map:
    print("No .xmp files found in", args.xmp_out)
    return

  copy_xmp_to_dng_dirs(xmp_map, args.lrdata)

if __name__ == "__main__":
  main()