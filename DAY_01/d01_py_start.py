word = "python"
print(word[:3])

listnum = [1,2,4,5,7,0]
arr = list(map(lambda x:x**2, listnum))
print(arr)

def doublestar(**k) :
    for x, y in k.items() :
        print(x)

doublestar(c1="Sita", c2="Sharma")

class NumBox :
    
    def __new__(cls, *args) :
        # if len(args) < 1 :
        #     return None
        # else :    
            print("__new__ method run")
            return super(NumBox, cls).__new__(cls)
            
    def __init__(slef, ms="__init__ method run") :
        slef.ms = ms
        # print(ms)

    def __repr__(self) :
        return str(self.ms)

    def printNum(self, num) :
        print(num)

nb = NumBox("__init__ caller")
# nb.printNum(1)
print(repr(nb))
print(str(nb))

