import model_SVC
import data_download
import dataset

#Just testing...
# download data
data_download.main()

# load preprocessed data
df = dataset.load_data()
