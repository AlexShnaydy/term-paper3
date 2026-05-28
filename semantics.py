import gensim
import urllib.request
import warnings
warnings.filterwarnings('ignore')

urllib.request.urlretrieve('http://vectors.nlpl.eu/repository/20/180.zip', 'ruscorpora_upos_cbow_300_20_2019.zip')

import zipfile
src = 'ruscorpora_upos_cbow_300_20_2019.zip'

with zipfile.ZipFile(src, 'r') as zip_ref:
  zip_ref.extractall('.')

m = 'model.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=True)

print(model.similarity('обычно_ADV', 'нормально_ADV'))


