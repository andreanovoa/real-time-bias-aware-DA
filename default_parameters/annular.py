from essentials.create import *
from essentials.models_physical import *
from essentials.Util import *
from essentials.bias_models import *
from essentials.plotResults import *
from essentials.run import *
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


