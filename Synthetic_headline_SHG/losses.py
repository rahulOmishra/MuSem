
class Losses(object):
    def __init__(self, div):
        self.div = div
        self.clear()

    def clear(self):
        self.loss, self.g, self.d, self.d_hz= 0.0, 0.0, 0.0, 0.0

    def add(self, loss, g, d,d_hz):
        self.loss += loss / self.div
        self.g += g / self.div
        self.d += d / self.div
        self.d_hz+=d_hz

    def output(self, s):
        print ('%s loss %.2f, g %.2f, d %.2f, adv %.2f' \
            % (s, self.loss, self.g, self.d,self.d_hz))
