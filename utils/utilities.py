import re

def convert_int(alist):
    """
    Convert all int-able string in list to int
    """
    for i in range(len(alist)):
        try:
            alist[i] = int(alist[i])
        except ValueError:
            pass
    return alist

prune_delimitor = re.compile("([\\s,.!?/\\'\"])")
prune_dict = {
    '',
    'and',
    'or'
}
def prune(sentence):
    """
    Prunng the empty and meaningless sentences
    """
    sentence = sentence.strip()
    tokens = prune_delimitor.split(sentence)
    l = 0
    r = len(tokens)
    while(l < r):
        if tokens[l].lower() not in prune_dict:
            break
        l += 2
    while(l < r):
        if tokens[r-1].lower() not in prune_dict:
            break
        r -= 2
    return ''.join(tokens[l: r])

def posstr(pos_list):
    """
    Stringify pos list
    """
    return ''.join(map(str, pos_list))
    
if __name__ == '__main__':
    def test_prune():
        print(prune("..and. ma,, wowo/ ,, . ., !")+"|")
        print(prune("ma,, wowo")+"|")
        print(prune("\tandma,, wowo")+"|")
        print(prune("and and ma,, wowo. ")+"|")
        print(prune("... ...")+"|")
    test_prune()
    