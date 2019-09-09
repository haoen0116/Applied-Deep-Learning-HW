wget https://www.dropbox.com/s/gyxpfwvknwxhpx3/BERT_model.bin
cp ./BERT_model.bin ./best/
cp ./BERT_model.bin ./strong/

wget https://www.dropbox.com/s/ksm7fs5m7mk41qm/9c41111e2de84547a463fd39217199738d1e3deb72d4fec4399e6e241983c6f0.ae3cef932725ca7a30cdcb93fc6e09150a55e2a130ec7af63975a16c153ae2ba

cp ./9c41111e2de84547a463fd39217199738d1e3deb72d4fec4399e6e241983c6f0.ae3cef932725ca7a30cdcb93fc6e09150a55e2a130ec7af63975a16c153ae2ba ./best/.pytorch_pretrained_bert/distributed_-1/

cp ./9c41111e2de84547a463fd39217199738d1e3deb72d4fec4399e6e241983c6f0.ae3cef932725ca7a30cdcb93fc6e09150a55e2a130ec7af63975a16c153ae2ba ./strong/.pytorch_pretrained_bert/distributed_-1/

# python3 bert_download.py
python3.7 -m spacy download en_core_web_sm
