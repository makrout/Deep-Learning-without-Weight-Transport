if [ "$1" == "bp" ]
then
        python main.py --dataset=mnist --algo=bp --n_epochs=400 --size_hidden_layers 500 --batch_size=128 --learning_rate=0.2 --test_frequency=1
        #python main.py --dataset=cifar10 --algo=bp --n_epochs=400 --size_hidden_layers 1000 --batch_size=128 --learning_rate=0.2 --test_frequency=1
elif [ "$1" == "fa" ]
then
        python main.py --dataset=mnist --algo=fa --n_epochs=400 --size_hidden_layers 500 --batch_size=128 --learning_rate=0.2 --test_frequency=1
        #python main.py --dataset=cifar10 --algo=fa --n_epochs=400 --size_hidden_layers 1000 --batch_size=128 --learning_rate=0.2 --test_frequency=1
elif [ "$1" == "wm" ]
then
        python main.py --dataset=mnist --algo=wm --n_epochs=400 --size_hidden_layers 500 --batch_size=128 --learning_rate=0.05 --test_frequency=1
        #python main.py --dataset=cifar10 --algo=wm --n_epochs=400 --size_hidden_layers 1000 --batch_size=128 --learning_rate=0.05 --test_frequency=1
elif [ "$1" == "kp" ]
then
        python main.py --dataset=mnist --algo=kp --n_epochs=400 --size_hidden_layers 500 --batch_size=128 --learning_rate=0.3 --test_frequency=1
        #python main.py --dataset=cifar10 --algo=kp --n_epochs=400 --size_hidden_layers 1000 --batch_size=128 --learning_rate=0.3 --test_frequency=1
else
	echo "Invalid input argument. Valid ones are either `bp`, `fa`, `wm` or `kp`."
	exit -1
fi
