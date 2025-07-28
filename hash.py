#!/usr/bin/env python

from ensure_pepper import ensure_pepper
ensure_pepper() 

import fire, hash_csvs

if __name__ == "__main__":
    fire.Fire(hash_csvs.hash_all_csvs)
