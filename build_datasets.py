#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for making datasets for the project
"""

import argparse
import logging
import os
import sys
import time


def main():
    """Main function"""

    logging.info("Building datasets")

    logging.info("Datasets built")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"log_time{time.time()}.log"),
        ],
    )
    logging.info("Starting main block.")

    parser = argparse.ArgumentParser(description="Build datasets for the project")

    args = parser.parse_args()
    logging.info(args)

    main()
