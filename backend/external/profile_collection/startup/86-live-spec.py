# Initialize the filename to today's date.
import time
from event_model import RunRouter
# from suitcase.specfile import Serializer

# to test the Serializer locally, Oct2022
import importlib
suitcaseSpecfile = importlib.import_module('85-suitcase-specfile')
Serializer = suitcaseSpecfile.Serializer


def spec_factory(name, doc):
    directory = "/nsls2/data/cms/legacy/xf11bm/ScanFiles"
    file_prefix = "cms_scan_" + time.strftime("%Y_%m_%d")
    
# # skip multiple motor scans
#     dims =doc.get('hints', {}).get('dimensions', [])
#     if len(dims) and (len(dims[0][0]) > 1):
#         return [], []

    # if doc.get('motors') != None: ## for test in the simulation mode
    #     if len(doc.get('motors',{})) > 1:
    #         return [], []

# Add plan names to this list to live export additional types of plans
    plan_alowed_list = {'scan', 'rel_scan', 'grid_scan' }
    if doc.get('plan_name', '') in plan_alowed_list:
        return [Serializer(directory, file_prefix=file_prefix, flush=True)], []
    else:
        return [], []


run_router = RunRouter([spec_factory])
RE.subscribe(run_router)


#run_router = RunRouter([spec_factory])
#RE.subscribe(run_router)
# NotImplementedError: The suitcase.specfile.Serializer is not designed to handle more than one descriptor.  If you need this functionality, please request it at https://github.com/NSLS-II/suitcase/issues. Until that time, this DocumentToSpec callback will raise a NotImplementedError if you try to use it with two event streams.

#this is run to install pymca
#conda create -n pymca_testing python
#conda activate pymca_testing
#conda install -c conda-forge pymca pyqt
# conda install -c conda-forge matplotlib
#pymca