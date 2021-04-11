
import data_download
import dataset

#Just testing...
# download data
data_download.main()

# load preprocessed data
df = dataset.load_data()

print(df.describe())
print(df.size)
print(df.head(10))