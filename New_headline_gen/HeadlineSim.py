from similarity import similarity

def headlineContent(file1, file2):
    head_pair_sim = dict()
    with open(file1) as f_original:
        ori_heads = f_original.readlines()

    with open(file2) as f_gen:
        generated_heads = f_gen.readlines()

    for i in range(len(ori_heads)):
        ori_head = ori_heads[i]
        gen_head = generated_heads[i]
        sim = similarity(ori_head,gen_head,info_content_norm=True)
        head_pair_sim[ori_head+'::'+gen_head] = sim
        print ori_head+''+gen_head+'\t'+str(sim)

    head_pair_sim_sort = sorted(head_pair_sim.iteritems(), key=lambda (k,v): (v,k),reverse=True)

    for key,val in head_pair_sim_sort:
        print "%s: %s" % (key, val)


def headlineSimilarity(headline1, headline2):

    print

if __name__ == '__main__':
    epoch = 100
    label = 1
    headlineContent('../data/train_head.'+str(label),'../tmp/clickbait_challenge.dev.epoch'+str(epoch)+'.'+str(label)+'.tsf')