ApresQuadPol
=========
This page contains the documentation for the ApresQuadPol class, which is the basic object in ImpDAR's apres processor ``qpdar``.
All of the files to define the class are in impdar/lib/ApresData, with the basic initialization and class properties found in __init__.py and addional functionality spread across _ApresDataSaving, _QuadPolDataProcessing.

.. autoclass:: impdar.lib.ApresData.ApresQuadPol
    :members: attrs_guaranteed, attrs_optional, shh, shv, svh, svv, range, decday, dt, snum, travel_time, lat, long, x_coord, y_coord, elev, thetas, HH, HV, VH, VV, chhvv, dphi_dz, check_attrs

.. automethod:: impdar.lib.ApresData.ApresQuadPol.save

.. automethod:: impdar.lib.ApresData.ApresQuadPol.rotational_transform

.. automethod:: impdar.lib.ApresData.ApresQuadPol.find_cpe

.. automethod:: impdar.lib.ApresData.ApresQuadPol.coherence2d

.. automethod:: impdar.lib.ApresData.ApresQuadPol.phase_gradient2d
