from data import *
from models import *
import argparse
import os
import pickle
import logging
import json

'''
PARAMETERS
'''
parser = argparse.ArgumentParser(description='NLI training')
parser.add_argument("--data_path", type=str, default='data', help="path to data")

# model
parser.add_argument("--encoder_type", type=str, default='GRUEncoder', help="see list of encoders")
parser.add_argument("--enc_hidden_dim", type=int, default=256, help="encoder nhid dimension")
parser.add_argument("--num_layer", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=256, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
parser.add_argument("--use_cuda", type=int, default=1, help="True or False")
parser.add_argument("--use_attention", type=int, default=0, help="use attentsion for final hidden state.")

# train
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0.2, help="classifier dropout")
parser.add_argument("--dpout_embed", type=float, default=0., help="embed dropout")
parser.add_argument("--embed_freeze", action='store_true', help="freeze embedding layer 0:False, 1:True")
parser.add_argument("--lr", type=float, default=0.0005, help="learning rate for adam")
parser.add_argument("--last_model", type=str, default="", help="train on last saved model")
parser.add_argument("--saved_model_name", type=str, default="model_try", help="saved model name")
parser.add_argument("--w2v_model", type=str, default="w2v-model.txt", help="w2v file name")
parser.add_argument("--weight_decay", type=float, default=0., help="L2 penalty")
parser.add_argument("--lr_decay_th", type=float, default=0., help="threshold on loss improve for learning rate decay")

params, _ = parser.parse_known_args()
print(params)


'''
SEED
'''
np.random.seed(10)
torch.manual_seed(10)


"""
DATA
"""
train, dev, test = get_question_pairs(params.data_path)
sentences = np.append(train['s1'], train['s2'])

if not os.path.exists(os.path.join(params.data_path,'word2ind.pickle')):
    word2ind, ind2word = get_word2ind_ind2word(sentences, min_n=5)
    with open( os.path.join(params.data_path, "word2ind.pickle" ), 'wb') as handle:
        pickle.dump(word2ind, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open( os.path.join(params.data_path, "ind2word.pickle" ), 'wb') as handle:
        pickle.dump(ind2word, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    print("Loading word2ind and ind2word ...")
    word2ind = pickle.load( open( os.path.join(params.data_path, "word2ind.pickle" ), "rb") )
    ind2word = pickle.load( open( os.path.join(params.data_path, "ind2word.pickle" ), "rb") )
    
word_embed_matrix = build_word_embed_matrix(word2ind, pretrained_wordVec=params.w2v_model)


'''
MODEL
'''
config_nli_model = {
    'n_words'        :  word_embed_matrix.shape[0],
    'word_emb_dim'   :  word_embed_matrix.shape[1],
    'enc_hidden_dim' :  params.enc_hidden_dim,
    'num_layer'      :  params.num_layer,
    'dpout_model'    :  params.dpout_model,
    'dpout_fc'       :  params.dpout_fc,
    'fc_dim'         :  params.fc_dim,
    'bsize'          :  params.batch_size,
    'n_classes'      :  params.n_classes,
    'pool_type'      :  params.pool_type,
    'encoder_type'   :  params.encoder_type,
    'use_cuda'       :  params.use_cuda==1,
    'dpout_embed'    :  params.dpout_embed,
    'embed_freeze'   :  params.embed_freeze,
    'embed_matrix'   :  word_embed_matrix,
    'weight_decay'   :  params.weight_decay,
    "use_attention"  : params.use_attention,
}
    

nli_net = NLINet(config_nli_model)
if params.last_model:
    print("load model {}".format(params.last_model))
    nli_net.load_state_dict(torch.load(os.path.join("saved_model", params.last_model, params.last_model)))
print(nli_net)

# loss 
weight = torch.FloatTensor(3).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)

# optimizer
from torch import optim
parameters = filter(lambda p: p.requires_grad, nli_net.parameters())
optimizer = optim.Adam(parameters, lr=params.lr, weight_decay=params.weight_decay)

# cuda 
if params.use_cuda:
    torch.cuda.manual_seed(10)
    torch.cuda.set_device(0)
    nli_net.cuda()
    loss_fn.cuda()


'''
TRAIN FUNCTION
'''
def adjust_learning_rate(optimizer):
    print("learning rate decay by half ... ")
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 2

def trainepoch(epoch):
    all_costs = []
    tot_costs = []
    logs = []
    correct = 0.0
    
    nli_net.train()
    permutation = np.random.permutation(len(train['s1']))
    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]
    
    for i in range(0, len(s1), params.batch_size):
        s1_batch, s1_len= get_inds_batch(s1[i: i+params.batch_size], word2ind)
        s2_batch, s2_len= get_inds_batch(s2[i: i+params.batch_size], word2ind)
        
        if params.use_cuda:
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(torch.LongTensor(target[i: i+params.batch_size])).cuda()
        else:
            s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
            tgt_batch = Variable(torch.LongTensor(target[i: i+params.batch_size]))
        k = s1_batch.size(1)
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        
        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.item())
        tot_costs.append(loss.item())
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if len(all_costs) == 100:
            logs.append('{0};  loss: {1};  accuracy train: {2}'.format(i, 
                            round(np.mean(all_costs), 3), round(100.*correct/(i+k), 3)))
            print(logs[-1])
            all_costs = []
            
    train_acc = round(100 * correct/len(s1), 3)
    train_loss = round(np.mean(tot_costs), 3)
    return train_loss, train_acc    

val_acc_best = -1e10
def evaluate(epoch, eval_type='dev',):
    nli_net.eval()
    correct = 0.0
    global val_acc_best
    s1 = dev['s1'] if eval_type == 'dev' else test['s1']
    s2 = dev['s2'] if eval_type == 'dev' else test['s2']
    target = dev['label'] if eval_type == 'dev' else test['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len= get_inds_batch(s1[i: i+params.batch_size], word2ind)
        s2_batch, s2_len= get_inds_batch(s2[i: i+params.batch_size], word2ind)
        
        if params.use_cuda:
            s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
            tgt_batch = Variable(torch.LongTensor(target[i: i+params.batch_size])).cuda()
        else:
            s1_batch, s2_batch = Variable(s1_batch), Variable(s2_batch)
            tgt_batch = Variable(torch.LongTensor(target[i: i+params.batch_size]))
            
        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

    # save model
    eval_acc = round(100 * correct / len(s1), 3)
    print('togrep:  results: epoch {0};  mean accuracy {1}:{2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'dev' and eval_acc > val_acc_best:
        print('saving model at epoch {0}'.format(epoch))      
        torch.save(nli_net.state_dict(), os.path.join(saved_folder, params.saved_model_name))
        val_acc_best = eval_acc

    return eval_acc


"""
Train model 
"""
saved_folder = os.path.join("saved_model", params.saved_model_name)        
if not os.path.exists(saved_folder): os.makedirs(saved_folder)
    
# with open( os.path.join(saved_folder, "config.pickle" ), 'wb') as handle:
#     pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)  
with open( os.path.join(saved_folder,'config.json'), 'w') as fp:
    json.dump( vars(params), fp)
    
logger = logging.getLogger(params.saved_model_name)
hdlr = logging.FileHandler( os.path.join(saved_folder, "train_process.log") )
formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

### TRAINING 

train_loss_ls = []
train_acc_ls = []
eval_acc_ls = []
eval_acc = 0
prev_loss = float('inf')

for i in range(params.n_epochs):
    print('\nTRAINING : Epoch ' + str(i) + " --- "+ params.saved_model_name)
    train_loss, train_acc = trainepoch(i)
    train_loss_ls.append(train_loss)
    train_acc_ls.append(train_acc)
    
    print("-"*100)
    print('\nTRAINING : Epoch ' + str(i) + " --- "+ params.saved_model_name)
    for pi in range(len(train_loss_ls)):
        train_result = 'results: epoch {0};  loss: {1};  mean accuracy train: {2}'.format(pi, train_loss_ls[pi], train_acc_ls[pi])
        print(train_result)
    logger.info(train_result)
    print("-"*100)

    if prev_loss-train_loss<params.lr_decay_th:
        adjust_learning_rate(optimizer)
    prev_loss = train_loss    
        
    if i%1==0:
        print("-"*100)
        print('\nEVALIDATING: Epoch ' + str(i) + " --- "+ params.saved_model_name)
        eval_acc = evaluate(i, eval_type='dev')
        eval_acc_ls.append(eval_acc)
        
        for pi in range(len(train_loss_ls)):
            dev_result = 'results: epoch {0};  mean accuracy dev: {1}'.format(pi, eval_acc_ls[pi])
            print(dev_result)
        logger.info(dev_result)
        print("-"*100)



