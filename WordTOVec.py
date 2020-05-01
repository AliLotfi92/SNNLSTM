
import string
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot



def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean_doc(doc):
    doc = doc.replace('--', ' ')
    tokens = doc.split()
    print(tokens)
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens



def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# Change the directory for
in_filename = 'Data/WordLevel.txt'
doc = load_doc(in_filename)


tokens = clean_doc(doc)

# length of window
length = 10+ 1
sequences = list()

for i in range(length, len(tokens)):
    seq = tokens[i - length:i]
    sequences.append(seq)

print('Total Sequences: %d' % len(sequences))

model = Word2Vec(sequences, size=100, min_count=1, window=10, iter=1000)
y = model['i']

pretrained_weights = model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape


X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
# Save the Word2Vec
model.save('Data/modelword.bin')
