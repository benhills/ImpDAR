#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 David Lilien <david.lilien@umanitoba.ca>
#
# Distributed under terms of the GNU GPL3.0 license.

"""
Make an executable for single actions of ApRES handling.
"""
import sys
import os.path
import argparse

from impdar.lib.ApresData import load_apres


def _get_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Choose a processing step')

    # Load and exit
    parser_load = _add_procparser(subparsers,
                                  'load',
                                  'load apres data',
                                  lambda x: x,
                                  defname='load')
    _add_def_args(parser_load)

    # Convert range
    parser_range = _add_procparser(subparsers,
                                   'range',
                                   'Range-convert apres data',
                                   apres_range,
                                   defname='ranged')
    parser_range.add_argument('pad',
                              help='Padding for fft',
                              type=int)
    parser_range.add_argument('--max_range',
                              default=4000.0,
                              help='Maximum range to process',
                              type=float)
    parser_range.add_argument('--winfun',
                              default='blackman',
                              help='Window type',
                              type=str,
                              choices=['blackman', 'bartlett', 'hamming', 'hanning', 'kaiser'])
    _add_def_args(parser_range)
    return parser


def _add_simple_procparser(subparsers, name, helpstr, func, defname='proc'):
    """Add a sub parser that can do a simple thing with no arguments."""
    parser = _add_procparser(subparsers, name, helpstr, func, defname=defname)
    _add_def_args(parser)
    return parser


def _add_procparser(subparsers, name, helpstr, func, defname='proc'):
    """Wrap adding subparser because we mostly want the same args."""
    parser = subparsers.add_parser(name, help=helpstr)
    parser.set_defaults(func=func, name=defname)
    return parser


def _add_def_args(parser):
    """Set some default arguments common to the different processing types."""
    parser.add_argument('fns',
                        type=str,
                        nargs='+',
                        help='The files to process')
    parser.add_argument('-o',
                        type=str,
                        help='Output to this file (folder if multiple inputs)')


def apres_range(dat, pad, max_range=4000, winfun='blackman', **kwargs):
    return dat.apres_range(pad, max_range=max_range, winfun=winfun)


def main():
    """Get arguments, process data, handle saving."""
    parser = _get_args()
    args = parser.parse_args(sys.argv[1:])
    if not hasattr(args, 'func'):
        parser.parse_args(['-h'])

    apres_data = [load_apres.load_apres_single_file(fn) for fn in args.fns]

    if args.name == 'load':
        pass
    else:
        for dat in apres_data:
            args.func(dat, **vars(args))

    if args.name == 'load':
        name = 'raw'
    else:
        name = args.name

    if args.o is not None:
        if ((len(apres_data) > 1) or (args.o[-1] == '/')):
            for d, f in zip(apres_data, args.fns):
                bn = os.path.split(os.path.splitext(f)[0])[1]
                if bn[-4:] == '_raw':
                    bn = bn[:-4]
                out_fn = os.path.join(args.o, bn + '_{:s}.h5'.format(name))
                d.save(out_fn)
        else:
            out_fn = args.o
            apres_data[0].save(out_fn)
    else:
        for d, f in zip(apres_data, args.fns):
            bn = os.path.splitext(f)[0]
            if bn[-4:] == '_raw':
                bn = bn[:-4]
            out_fn = bn + '_{:s}.h5'.format(name)
            d.save(out_fn)


def crop(dat, lim=0, top_or_bottom='top', dimension='snum', **kwargs):
    """Crop in the vertical."""
    dat.crop(lim, top_or_bottom=top_or_bottom, dimension=dimension)


if __name__ == '__main__':
    main()