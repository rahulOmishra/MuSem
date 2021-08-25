
class Losses(object):
    def __init__(self, div):
        self.div = div
        self.clear()

    def clear(self):
        self.g = 0.0

    def add(self, g):
        self.g += g / self.div

    def output(self, s):
        print ('%s loss g %.2f' \
            % (s, self.g))
