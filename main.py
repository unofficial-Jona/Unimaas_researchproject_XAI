import data_download
import dataset

import model_SVC
import model_random_forest

#Just testing...
# download data
data_download.main()

# load preprocessed data
df = dataset.load_data()
