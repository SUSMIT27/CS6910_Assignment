# CS6910_Assignment <br />
MultiClassClassification<br />
to run train.py file <b>"names a bit different" </b><br />
parser = argparse.ArgumentParser()<br />
parser.add_argument('-wp' , '--wandb_project', help='Project name used to track experiments in Weights & Biases dashboard' , type=str, default='myprojectname')<br />
parser.add_argument('-we', '--wandb_entity' , help='Wandb Entity used to track experiments in the Weights & Biases dashboard.' , type=str, default='myname')<br />
parser.add_argument('-d', '--dataset', help='choices:<b> ["mnist", "fashion_mnist"]'</b>, type=str, default='fashion_mnist')<br />
parser.add_argument('-e', '--epochs', help="Number of epochs to train neural network.", type=int, default=5)<br />
parser.add_argument('-b', '--batch_size', help="Batch size used to train neural network.", type=int, default=32)<br />
parser.add_argument('-l','--loss', help = 'choices: <b>["square_error_loss", "cross_entropy"]'</b> , type=str, default='cross_entropy')<br />
parser.add_argument('-o', '--optimizer', help = 'choices:<b> ["gd", "momentum", "NAG", "RMSProp", "Adam", "Nadam"]'</b>, type=str, default = 'gd')<br />
parser.add_argument('-lr', '--learning_rate', help = 'Learning rate used to optimize model parameters', type=float, default=0.005)<br />
parser.add_argument('-m', '--momentum', help='Momentum used by momentum and nag optimizers.',type=float, default=0.5)<br />
parser.add_argument('-beta', '--beta', help='Beta used by rmsprop optimizer',type=float, default=0.5)<br />
parser.add_argument('-beta1', '--beta1', help='Beta1 used by adam and nadam optimizers.',type=float, default=0.5)<br />
parser.add_argument('-beta2', '--beta2', help='Beta2 used by adam and nadam optimizers.',type=float, default=0.5)<br />
parser.add_argument('-eps', '--epsilon', help='Epsilon used by optimizers.',type=float, default=0.000001)<br />
parser.add_argument('-w_d', '--weight_decay', help='Weight decay used by optimizers.',type=float, default=.0)<br />
parser.add_argument('-w_i', '--weight_init', help = 'choices: <b>["random", "xavier"]'</b>, type=str, default='random')<br />
parser.add_argument('-nhl', '--num_layers', help='Number of hidden layers used in feedforward neural network.',type=int, default=3)<br />
parser.add_argument('-sz', '--hidden_size', help ='Number of hidden neurons in a feedforward layer.', nargs='+', type=int, default=32, required=False)<br />
parser.add_argument('-a', '--activation', help='choices:<b> ["sigmoid", "tanh", "relu"]'</b>, type=str, default='tanh')<br />
parser.add_argument('--hlayer_size', type=int, default=32)<br />
parser.add_argument('-oa', '--output_activation', help = 'choices:<b> ["softmax"]'</b>, type=str, default='softmax')<br />
parser.add_argument('-oc', '--output_size', help ='Number of neurons in output layer used in feedforward neural network.', type = int, default = 10)<br />

name of my project in wandb is <b> assignment1_CS910 </b>

File CS6910_Assignment(1-10).ipynb is the main file and all Questions are present there.
train.py is for checking purpose for argparse question
