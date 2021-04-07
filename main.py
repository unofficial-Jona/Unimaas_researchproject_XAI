
import data_download
import dataset


# download data
data_download.main()

# load preprocessed data
df = dataset.load_data()

