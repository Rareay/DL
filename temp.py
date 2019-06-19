import tensorflow as tf

class one():
    def __init__(self):
        self.a = 1
    def fun(self):
        print("one")

class tow():
    def __init__(self):
        self.a = 1
    def fun(self):
        ONE = one()
        ONE.fun()

TOW = tow()
TOW.fun()
