Guide:

The env is in Python 3.7.2

Please download the model:
1. bash download.sh

If you want to do [simple.sh] please type the command below:
2. bash simple.sh

If you want to print the graph, you should modify your code.
	Let [base_trainer.py def _run_epoch(self, mode):] return the acc value [return self._stat[mode]['acc(label)']]
	Let [def start(self):] return [train_acc, eval_acc] through append the historical data [train_acc.append(self._run_epoch('train')), eval_acc.append(self._run_epoch('eval'))]
	Finally, plot the graph through the matplotlib
	
	code:
		plt.plot(range(10), [1 - i for i in train_acc], range(10), [1 - i for i in eval_acc])
		plt.legend(['Training perplexity', 'Validation perplexity'], loc='upper left')
		plt.xlabel('Epoch')
		plt.ylabel('Perplexity')
		plt.title('Perplexity from training and validation')
		plt.show()