import time, random, re

def replace4( sentences ):
    pd = patterns_dict.get
    def repl(m):
        w = m.group()
        return pd(w,w)
    
    for n, sentence in enumerate( sentences ):
        sentence = re.sub(r"\w+", repl, sentence)

# Build test set
test_words = [ ("word%d" % _) for _ in range(50000) ]
test_sentences = [ " ".join( random.sample( test_words, 10 )) for _ in range(1000) ]

# Create search and replace patterns
patterns = [ (("word%d" % _), ("repl%d" % _)) for _ in range(20000) ]
patterns_dict = dict( patterns )
# patterns_comp = [ (re.compile("\\b"+search+"\\b"), repl) for search, repl in patterns ]


def test( func, num ):
    t = time.time()
    func( test_sentences[:num] )
    print( "%30s: %.02f sentences/s" % (func.__name__, num/(time.time()-t)))

print( "Sentences", len(test_sentences) )
print( "Words    ", len(test_words) )

test( replace4, 1000 )