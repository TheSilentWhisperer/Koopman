# How to run the scripts ?

## Training the auto-encoder
python3 train_autoencoder.py

## Training the predictor
python3 train_predictor.py --batch_size 8 --k 1
python3 train_predictor.py --batch_size 8 --k 4
python3 train_predictor.py --batch_size 8 --k 8
python3 train_predictor.py --batch_size 8 --k 16
python3 train_predictor.py --batch_size 8 --k 32
python3 train_predictor.py --batch_size 8 --k 50

## Training everything together
python3 train_full_model.py --batch_size 8 --k 50
