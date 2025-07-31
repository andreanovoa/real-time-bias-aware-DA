from src.create import *
from src.models_physical import *
from src.utils import *
from src.bias import *
from src.plot_results import *
from src.run import *
# ============================================================================================== #

path_dir = os.path.realpath(__file__).split('main')[0]
#
if os.path.isdir('/mscott/'):
    data_folder = '/mscott/an553/data/'  # set working directory to mscott
    # os.chdir(data_folder)  # set working directory to mscott
else:
    data_folder = "data/"


folder = 'results/Annular/'
figs_dir = folder + 'figs/'
out_dir = folder+"/out/"

os.makedirs(figs_dir, exist_ok=True)


