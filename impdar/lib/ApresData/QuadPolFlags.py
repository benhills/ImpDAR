#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 David Lilien <dlilien90@gmail.com>
#
# Distributed under terms of the GNU GPL3.0 license.

"""
Flags to keep track of processing steps
"""

import numpy as np
import h5py


class QuadPolFlags():
    """Flags that indicate the processing that has been used on the data.

    These are used for figuring out whether different processing steps have been performed. They also contain some information about the input arguments for some (but not all) of the processing steps.

    Attributes
    ----------
    batch: bool
        Legacy indication of whether we are batch processing. Always False.
     """

    def __init__(self):
        self.rotation = np.zeros((2,))
        self.coherence = np.zeros((3,))
        self.phasegradient = False
        self.attrs = ['rotation','coherence','phasegradient']
        self.attr_dims = [2,3,None]

    def read_h5(self, grp):
        subgrp = grp['QuadPolFlags']
        for attr in subgrp.attrs.keys():
            val = subgrp.attrs[attr]
            if isinstance(val, h5py.Empty):
                val = None
            setattr(self, attr, val)

    def write_h5(self, grp):
        """Write to a subgroup in hdf5 file

        Parameters
        ----------
        grp: h5py.Group
            The group to which the ApresFlags subgroup is written
        """
        subgrp = grp.create_group('QuadPolFlags')
        for attr in self.attrs:
            val = getattr(self, attr)
            if val is None:
                subgrp[attr] = h5py.Empty('f')
            else:
                if hasattr(val, 'dtype'):
                    val = val.astype('f')
                subgrp.attrs[attr] = val

    def to_matlab(self):
        """Convert all associated attributes into a dictionary formatted for use with :func:`scipy.io.savemat`
        """
        outmat = {att: getattr(self, att) for att in self.attrs}
        return outmat

    def from_matlab(self, matlab_struct):
        """Associate all values from an incoming .mat file (i.e. a dictionary from :func:`scipy.io.loadmat`) with appropriate attributes
        """
        for attr, attr_dim in zip(self.attrs, self.attr_dims):
            setattr(self, attr, matlab_struct[attr][0][0][0])
            # Use this because matlab inputs may have zeros for flags that
            # were lazily appended to be arrays, but we preallocate
            if attr_dim is not None and getattr(self, attr).shape[0] == 1:
                setattr(self, attr, np.zeros((attr_dim, )))