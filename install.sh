pip install -r requirements.txt

mkdir -p extracted
mkdir -p embeddings
mkdir -p dataset/seqlabel
mkdir -p dataset/relation
mkdir -p checkpoints

cd embeddings

if [ ! -f glove.840B.300d.txt ]; then
    echo "downloading Glove 840B 300d"
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip glove.840B.300d.zip
fi

cd ..

git submodule init
git submodule update

cd models/pytorch-pretrained-BERT
pip install --editable ./

echo "install finished!"