Modéliser les consommations électriques de sites B2B
----------
My solution to the challenge  `Datascience.net <http://www.datascience.net/fr/home/>`_::


Dependencies
============ 
- pandas (pip install -U pandas)
- numpy (pip install -U numpy)
- sklearn (pip install -U sklearn)
- (matplotlib)


Running and testing
===================

To test and run the implementation, use the driver `main`::

   ./main --verbose --use_cache_data --use_cache_trainingset --test --plot --compute_reality

A file named str(time.time())+'_ML.csv' will be created in ./results/ if istest == False.
Else, the training is not done on the last ten days, and a comparison with the true value is done.
For testing purpose, running::

   %run main

in IPython, and play around with matplotlib is the best solution.


TODO LIST
=========

* nothing
