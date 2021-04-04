import sys
import time
from multiprocessing import Process, Manager, get_context

sys.path.insert(1, "../")
import facenet

class Thread():
    def __init__(self, name_list, classifier_list):
        self.name_list = name_list
        self.classifier_list = classifier_list
        self.manager = Manager()


    def get_best_match(self, name, classifier, embed_list):
        start = time.time()
        for nonce_pos, embed in enumerate(embed_list):
            if embed is not None:
                best_match = classifier.predict(embed)[0]
                self.best_matches["{}_{}".format(name, nonce_pos)] = best_match
            else:
                self.best_matches["{}_{}".format(name, nonce_pos)] = "ryan_park"
        print("INDIVIDUAL {}".format(time.time()-start))
        

    def run(self, embed_list):
        self.best_matches = self.manager.dict()
        start = time.time()
        embed_list = embed_list + embed_list
        p1 = Process(target=self.get_best_match, args=(self.name_list[0], self.classifier_list[0], embed_list,))
        p2 = Process(target=self.get_best_match, args=(self.name_list[1], self.classifier_list[1], embed_list,))
        print("INSTANTIATE {}".format(time.time()-start))
        start = time.time()
        p1.start()
        p2.start()
        print("START {}".format(time.time()-start))
        start = time.time()
        p1.join()
        p2.join()
        print("JOIN {}".format(time.time()-start))
        return self.best_matches
        

