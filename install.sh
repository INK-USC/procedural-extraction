pip install -r requirements.txt

mkdir -p logs
mkdir -p extracted
mkdir -p embeddings
mkdir -p dataset/seqlabel
mkdir -p dataset/relation

cd embeddings

if [ ! -f glove.840B.300d.txt ]; then
    echo "downloading Glove 840B 300d"
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip glove.840B.300d.zip
fi

cd ..
echo "install finished!"