import os

extracted = lambda dir_extracted, datasetid: os.path.join(dir_extracted, '%02d.tgt.extracted.pkl' % datasetid)

src = lambda dir_data, datasetid: os.path.join(dir_data, '%02d.src.txt' % datasetid)
src_ref = lambda dir_data, datasetid: os.path.join(dir_data, '%02d.src.ref.txt' %  datasetid)
tgt = lambda dir_data, datasetid: os.path.join(dir_data, '%02d.tgt.txt' % datasetid)