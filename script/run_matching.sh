 mkdir -p extracted/embavg
 ds='1 2 3 4 6'
 for d in $ds
 do
    python extract_samples.py $d embavg \
    --dir_extracted extracted/embavg
done