#!/usr/bin/env python3
import asyncio
import re
from collections import defaultdict

from caproto import ChannelType
from caproto.server import ioc_arg_parser, run, pvproperty, PVGroup




def main():
    _, run_options = ioc_arg_parser(
        default_prefix='XF:11BM-ES',
        desc="Shutter IOC")
    run_options['interfaces'] = ['127.0.0.1']
    shutter = Shutter()
    run(shutter.pvdb,
        **run_options)


if __name__ == '__main__':
    main()