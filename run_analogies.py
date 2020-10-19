import numpy as np
import pickle
import argparse
import re
from lemmagen3 import Lemmatizer

class Analogy:
    def __init__(self, line, n):
        stats = line.split('\t')
        self.poklicm1 = stats[2]
        self.poklicm2 = stats[4]
        self.countm1 = int(stats[3]) if stats[3] else 0
        self.countm2 = int(stats[5]) if stats[5] else 0
        self.poklicf1 = stats[7]
        self.poklicf2 = stats[9]
        self.countf1 = int(stats[8]) if stats[8] else 0
        self.countf2 = int(stats[10]) if stats[10] else 0
        self.m_in_f_candidates = ['N/A',]*n
        self.f_in_m_candidates = ['N/A',]*n
        self.f_candidates_dist = ['NaN',]*n
        self.m_candidates_dist = ['NaN',]*n
        self.Moutputrank = ['', '-1']
        self.Foutputrank = ['', '-1']
        
    def m_input(self, word2id, embmatrix, id_to_word, n, mf_vectors, lem):
        #a = embmatrix[word2id['moški']]
        #c = embmatrix[word2id['ženska']]
        a = mf_vectors[0]
        c = mf_vectors[1]
        
        if self.countm1 >= self.countm2:
            bword = self.poklicm1
        else:
            bword = self.poklicm2
        if bword in word2id:
            b = embmatrix[word2id[bword]]
            d = b-a+c
            d /= np.sqrt(d.dot(d))
            self.m_in_f_candidates, self.f_candidates_dist = closest_words(d, embmatrix, id_to_word, n, lem)
            self.Moutputrank[0] = bword
            self.Moutputrank[1] = rankAword(d, embmatrix, word2id[bword])
        else:
            self.Moutputrank[0] = 'N/A:'+bword
        
    def f_input(self, word2id, embmatrix, id_to_word, n, mf_vectors, lem):
        #a = embmatrix[word2id['ženska']]
        #c = embmatrix[word2id['moški']]
        a = mf_vectors[1]
        c = mf_vectors[0]
        
        if self.countf1 >= self.countf2:
            bword = self.poklicf1
        else:
            bword = self.poklicf2
        if bword in word2id:
            b = embmatrix[word2id[bword]]
            d = b-a+c
            d /= np.sqrt(d.dot(d))
            self.f_in_m_candidates, self.m_candidates_dist = closest_words(d, embmatrix, id_to_word, n, lem)
            self.Foutputrank[0] = bword
            self.Foutputrank[1] = rankAword(d, embmatrix, word2id[bword])
        else:
            self.Foutputrank[0] = 'N/A:'+bword
            
    def filter_female(self):
        f_filtered = []
        for c in self.m_in_f_candidates:
            if c not in [self.poklicm1, self.poklicm2, 'moški', 'ženska'] and c not in f_filtered:
                f_filtered.append(c)
        return f_filtered
    
    def filter_male(self):
        #f_candidates_filtered =  list(filter(lambda x: x not in [e.poklicm1, e.poklicm2, 'moški', 'ženska'], e.m_in_f_candidates))
        m_filtered = []
        for c in self.f_in_m_candidates:
            if c not in [self.poklicf1, self.poklicf2, 'moški', 'ženska'] and c not in m_filtered:
                m_filtered.append(c)
        return m_filtered
            
        
def closest_words(vector, embmatrix, id_to_word, n, lem):
    # embmatrix = np.array([a,b,c,d,e,f,g...]), where a,b,c,d... are word vectors
    dotproduct = embmatrix.dot(vector)
    maxn_indices = dotproduct.argsort()[-n:][::-1]
    if lem:
        words = [lemmatize(lem,id_to_word[int(id)].lower()) for id in maxn_indices]
    else:
        words = [id_to_word[int(id)].lower() for id in maxn_indices]
    wordsvectors = np.array([embmatrix[int(id)] for id in maxn_indices])
    similarities = wordsvectors.dot(vector)
    return words, similarities

def rankAword(vector, embmatrix, idtorank):
    dotproduct = embmatrix.dot(vector)
    indices = list(dotproduct.argsort()[::-1])
    rank = indices.index(idtorank) #min rank = 0 (best)
    return str(rank+1)

def lemmatize(nlp, word):
    #return nlp(word).sentences[0].words[0].lemma
    return nlp.lemmatize(word)

def load_emb(embfile):
    picklename = 'embpickle/'+embfile.split('/')[-1]+'.samocrke.pickle'
    try:
        with open(picklename, 'rb') as f:
            word_to_id, id_to_word, embmatrix = pickle.load(f)
    except:
        word_to_id = {}
        id_to_word = []
        embmatrix = []
        idcount = 0
        necrke = re.compile('(\W|\d)')
        with open(embfile, 'r') as reader:
            header = reader.readline()
            dim = int(header.split()[1])
            for line in reader:
                line = line.split()
                word = line[0]
                if necrke.search(word):
                    continue
                id_to_word.append(word)
                word_to_id[word] = idcount
                try:
                    vector = np.asarray([float(v) for v in line[1:]])
                    vector = vector/np.sqrt(vector.dot(vector))
                except:
                    continue
                if len(vector) != dim:
                    continue
                else:
                    embmatrix.append(vector)
                    idcount += 1
        with open(picklename, 'wb') as f:
            pickle.dump([word_to_id, id_to_word, embmatrix], f, pickle.HIGHEST_PROTOCOL)
    return id_to_word, word_to_id, np.asarray(embmatrix)
  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--embeddings', required=True, help="Path to embeddings file.")
    parser.add_argument('-n', '--n', type=int, default=25, help="Search among top n candidates.")
    parser.add_argument('-o', '--output', required=True, help="Output prefix.")
    parser.add_argument('-i', '--input', default="Poklici_enobesedni.csv", help="Analogies .tsv file")
    parser.add_argument('-l', '--lemmatize', action="store_true", help="Lemmatize the embeddings.")
    parser.add_argument('--avginput', action="store_true", help="Take averages of male and female vectors as input instead of just words 'man' and 'woman'.")
    args = parser.parse_args()
    entries = []
    if args.lemmatize:
        nlp = Lemmatizer('sl')
    else:
        nlp = False
        
    with open(args.input, 'r') as reader:
        reader.readline()
        for line in reader:
            entries.append(Analogy(line, args.n))
    id2word, word2id, embmatrix = load_emb(args.embeddings)
    
    m_vector = embmatrix[word2id['moški']]
    f_vector = embmatrix[word2id['ženska']]
    if args.avginput:
        genderpairs = [('gospod', 'gospa'), ('fant', 'dekle'), ('fant', 'punca'), ('deček', 'deklica'), ('brat', 'sestra'), ('oče', 'mati'), ('sin', 'hči'), ('dedek', 'babica'), ('mož', 'žena'), ('stric', 'teta'), ('on', 'ona')]
        for p in genderpairs:
            m_vector += embmatrix[word2id[p[0]]]
            f_vector += embmatrix[word2id[p[1]]]
        m_vector /= (len(genderpairs)+1)
        f_vector /= (len(genderpairs)+1)
  
    correct_m_input = {1: 0, 5: 0, 10: 0, 20: 0}
    correct_f_input = {1: 0, 5: 0, 10: 0, 20: 0}
    correct_m_input_filtered = {1: 0, 5: 0, 10: 0, 20: 0}
    correct_f_input_filtered = {1: 0, 5: 0, 10: 0, 20: 0}
    m_input_coverage = 0
    f_input_coverage = 0
    with open(args.output+'.1.csv', 'w') as writer1, open(args.output+'.2.csv', 'w') as writer2:
        writer1.write('PoklicM,KandidatŽ,rank_KŽ,cos_similarity\n')
        writer2.write('PoklicŽ,KandidatM,rank_KM,cos_similarity\n')
        for e in entries:
            #print(e.poklicm1, e.poklicf1, e.poklicf2, e.countf1, e.countf2)
            e.m_input(word2id, embmatrix, id2word, args.n, [m_vector, f_vector], nlp)
            e.f_input(word2id, embmatrix, id2word, args.n, [m_vector, f_vector], nlp)
            if e.m_in_f_candidates[0] != 'N/A':
                m_input_coverage += 1
            if e.f_in_m_candidates[0] != 'N/A':
                f_input_coverage += 1
            f_candidates_filtered = e.filter_female() # list(filter(lambda x: x not in [e.poklicm1, e.poklicm2, 'moški', 'ženska'], e.m_in_f_candidates))
            m_candidates_filtered = e.filter_male() #list(filter(lambda x: x not in [e.poklicf1, e.poklicf2, 'moški', 'ženska'], e.f_in_m_candidates))
            fcount = 0
            mcount = 0
            for c in f_candidates_filtered[:10]:
                j = e.m_in_f_candidates.index(c)
                writer1.write(e.poklicm1+','+c+','+str(j+1)+','+str(e.f_candidates_dist[j])+'\n')
            for c in m_candidates_filtered[:10]:
                j = e.f_in_m_candidates.index(c)
                writer2.write(e.poklicf1+','+c+','+str(j+1)+','+str(e.m_candidates_dist[j])+'\n')
            #for j in range(args.n):
            #    if e.m_in_f_candidates[j] in f_candidates_filtered and fcount < 10:
            #        writer1.write(e.poklicm1+','+e.m_in_f_candidates[j]+','+str(j+1)+','+str(e.f_candidates_dist[j])+'\n')
            #        #print(j)
            #        fcount += 1
            #    if e.f_in_m_candidates[j] in m_candidates_filtered and mcount < 10:
            #        writer2.write(e.poklicf1+','+e.f_in_m_candidates[j]+','+str(j+1)+','+str(e.m_candidates_dist[j])+'\n')
            #        #print(j)
            #        mcount += 1
            for i in [1, 5, 10, 20]:               
                if e.poklicf1 in e.m_in_f_candidates[:i] or e.poklicf2 in e.m_in_f_candidates[:i]:
                    correct_m_input[i] += 1
                if e.poklicm1 in e.f_in_m_candidates[:i] or e.poklicm2 in e.f_in_m_candidates[:i]:
                    correct_f_input[i] += 1
                if e.poklicf1 in f_candidates_filtered[:i] or e.poklicf2 in f_candidates_filtered[:i]:
                    correct_m_input_filtered[i] += 1
                if e.poklicm1 in m_candidates_filtered[:i] or e.poklicm2 in m_candidates_filtered[:i]:
                    correct_f_input_filtered[i] += 1
    
    with open(args.output+'.inputrank.csv', 'w') as writer:
        writer.write('PoklicVhod,rankPVnaIzhodu\n')
        for e in entries:
            writer.write(','.join(e.Moutputrank)+'\n')
            writer.write(','.join(e.Foutputrank)+'\n')
            
    with open(args.output+'.condensed.txt', 'w') as writer:
        writer.write('Coverage (kolikšen delež poklicev se pojavi v embeddingih):\n')
        writer.write('moški poklici: '+str(m_input_coverage/len(entries))+'\n')
        writer.write('ženski poklici: '+str(f_input_coverage/len(entries))+'\n')
        writer.write('\r\nAnalogy accuracy, all = uspešnost med vsemi, če se poklic ne pojavi se šteje za nepravilno določen, covered = samo med tistimi, ki se pojavijo, če se poklic ne pojavi, se ne upošteva.\n')
        writer.write('m input, f output: (all / covered)\n')
        for i in [1,5,10,20]:
            printstring = 'acc@'+str(i)+' = '+str(correct_m_input[i]/len(entries))+' / '+str(correct_m_input[i]/m_input_coverage)
            writer.write(printstring+'\n')
        writer.write('\nf input, m output: (all / covered)\n')
        for i in [1,5,10,20]:
            printstring = 'acc@'+str(i)+' = '+str(correct_f_input[i]/len(entries))+' / '+str(correct_f_input[i]/f_input_coverage)
            writer.write(printstring+'\n')
        writer.write('\nm input, f output, filtered: (all / covered)\n')
        for i in [1,5,10,20]:
            printstring = 'acc@'+str(i)+' = '+str(correct_m_input_filtered[i]/len(entries))+' / '+str(correct_m_input_filtered[i]/m_input_coverage)
            writer.write(printstring+'\n')
        writer.write('\nf input, m output, filtered: (all / covered)\n')
        for i in [1,5,10,20]:
            printstring = 'acc@'+str(i)+' = '+str(correct_f_input_filtered[i]/len(entries))+' / '+str(correct_f_input_filtered[i]/f_input_coverage)
            writer.write(printstring+'\n')
    
    
if __name__ == "__main__":
    main()
