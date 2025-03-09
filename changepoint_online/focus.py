import math,sys

################
##  Families  ##
################

class CompFunc:

    def __init__(self, st, tau, theta0):
        self.st = st
        self.tau = tau
        self.theta0 = theta0



class UnivariateCompFunc(CompFunc):

    def __init__(self, st, tau, m0, theta0):
        super().__init__(st, tau, theta0)
        self.m0 = m0


    def argmax(self, cs):

        return (cs.sn - self.st) / (cs.n - self.tau)

    def get_max(self, cs):

        return self.eval(self.argmax(cs), cs)


class GaussianClass(UnivariateCompFunc):
    """
    This function represents a Gaussian component function. For more details, see `help(CompFunc)`.
    """
    def eval(self, x, cs):
        c = cs.n - self.tau
        s = cs.sn - self.st

        if self.theta0 is None:
            return -0.5 * c * x ** 2 + s * x + self.m0
        else:
            out = c * x ** 2 - 2 * s * x - (c * self.theta0 ** 2 - 2 * s * self.theta0)
            return -out / 2
        
def Gaussian(loc=None):

    return lambda st, tau, m0: GaussianClass(st, tau, m0, loc)
    
class BernoulliClass(UnivariateCompFunc):
    def eval(self, x, cs):
        c = cs.n - self.tau
        s = cs.sn - self.st
        if self.theta0 is None:
            return s * math.log(x) + (c - s) * math.log(1 - x) + self.m0
        else:
            return s * math.log(x / self.theta0) + (c - s) * math.log((1 - x) / (1 - self.theta0))
        
    def argmax(self, cs):
        agm = (cs.sn - self.st) / (cs.n - self.tau)
        if agm == 0:
            #return sys.float_info.min # this does not work
            return 1e-9
        elif agm == 1:
            #return 1 - sys.float_info.min # this does not work
            return 1 - 1e-9
        else:
            return agm
        
def Bernoulli(p=None):

    return lambda st, tau, m0 : BernoulliClass(st, tau, m0, p)


class PoissonClass(UnivariateCompFunc):
    def eval(self, x, cs):
        c = cs.n - self.tau
        s = cs.sn - self.st
        if self.theta0 is None:
            return -c * x + s * math.log(x) + self.m0
        else:
            return -c * (x - self.theta0) + s * math.log(x / self.theta0)
        
    def argmax(self, cs):
        agm = (cs.sn - self.st) / (cs.n - self.tau)
        return agm if agm != 0 else sys.float_info.min

def Poisson(lam=None):

    return lambda st, tau, m0 : PoissonClass(st, tau, m0, lam)

class GammaClass(UnivariateCompFunc):
    def __init__(self, st, tau, m0, theta0, shape):
        super().__init__(st, tau, m0, theta0)
        self.shape = shape

    def eval(self, x, cs):
        c = cs.n - self.tau
        s = cs.sn - self.st
        if self.theta0 is None:
            return -c * self.shape * math.log(x) - s * (1 / x) + self.m0
        else:
            return c * self.shape * math.log(self.theta0 / x) - s * (1 / x - 1 / self.theta0)
        
    def argmax(self, cs):
        return (cs.sn - self.st) / (self.shape * (cs.n - self.tau))

def Gamma(rate=None, scale=None, shape=1):

    if rate is not None:
        if scale is not None:
            raise ValueError("You can only provide either 'rate' or 'scale', not both.")
        else:
            scale = 1 / rate

    return lambda st, tau, m0: GammaClass(st, tau, m0, scale, shape)

def Exponential(rate=None):

    return Gamma(rate=rate, shape=1)


################################
##########   FOCUS   ###########
################################


class Focus:

    def __init__(self, comp_func, side = "both") :

        self.cs = Focus._CUSUM()
        self.ql = Focus._Cost(ps = [comp_func(0.0, 0, 0.0)])
        self.qr = Focus._Cost(ps = [comp_func(0.0, 0, 0.0)])
        self.comp_func = comp_func
        self.side = side

    def statistic(self) :
 
        return max(self.ql.opt, self.qr.opt)

    def changepoint(self) :

        def _argmax(x) :
            return max(zip(x,range(len(x))))[1]
        if self.ql.opt > self.qr.opt:
            i = _argmax([p.get_max(self.cs) - 0.0 for p in self.ql.ps[:-1]])
            most_likely_changepoint_location = self.ql.ps[i].tau
        else:
            i = _argmax([p.get_max(self.cs) - 0.0 for p in self.qr.ps[:-1]])
            most_likely_changepoint_location = self.qr.ps[i].tau
        return {"stopping_time": self.cs.n,"changepoint": most_likely_changepoint_location}
        
    def update(self, y):

        # updating the cusums and count with the new point
        self.cs.n += 1
        self.cs.sn += y

        # updating the value of the max of the null (for pre-change mean unkown)
        m0 = 0
        if self.qr.ps[0].theta0 is None:
            m0 = self.qr.ps[0].get_max(self.cs)

        if self.side not in  ["left", "right", "both"]:
            raise ValueError("size should be either 'both', 'right' or 'left'.")

        if self.side == "both" or self.side == "right":
            Focus._prune(self.qr, self.cs, "right")  # true for the right pruning
            self.qr.opt = Focus._get_max_all(self.qr, self.cs, m0)
            # add a new point
            self.qr.ps.append(self.comp_func(self.cs.sn, self.cs.n, m0))
        if self.side == "both" or self.side == "left":
            Focus._prune(self.ql, self.cs, "left")  # false for the left pruning
            self.ql.opt = Focus._get_max_all(self.ql, self.cs, m0)
            self.ql.ps.append(self.comp_func(self.cs.sn, self.cs.n, m0))


    class _Cost:
        def __init__(self, ps, opt=-1.0):
            self.ps = ps  # a list containing the various pieces
            self.opt = opt  # the global optimum value for ease of access
    class _CUSUM:
        def __init__(self, sn=0, n=0):
            self.sn = sn
            self.n = n

    def _prune(q, cs, side="right"):
        i = len(q.ps)
        if i <= 1:
            return q
        if side == "right":
            def cond(q1, q2):
                return q1.argmax(cs) <= q2.argmax(cs)
        elif side == "left":
            def cond(q1, q2):
                return q1.argmax(cs) >= q2.argmax(cs)
        while cond(q.ps[i - 1], q.ps[i - 2]):
            i -= 1
            if i == 1:
                break
        q.ps = q.ps[:i]
        return q
    
    def _get_max_all(q, cs, m0):
        return max(p.get_max(cs) - m0 for p in q.ps)

    
class TruncatedFocus(Focus):
    
    def __init__(self, comp_func, side = "both", window_size=500):
        super().__init__(comp_func=comp_func, side=side)  # Call Parent's init
        self.window_size = window_size
        self.cs = TruncatedFocus._TruncatedCUSUM(window_size=window_size)  # Change attr2 in Child


    def _prune(q, cs, side="right"):
        i = len(q.ps)
        if i <= 1:
            return q
        if side == "right":
            def cond(q1, q2):
                return q1.argmax(cs) <= q2.argmax(cs)
        elif side == "left":
            def cond(q1, q2):
                return q1.argmax(cs) >= q2.argmax(cs)
        while cond(q.ps[i - 1], q.ps[i - 2]):
            i -= 1
            if i == 1:
                break

        if side == "right":
            q.ps = [p for p in q.ps if p.tau >= cs.total_n - cs.n]
        elif side == "left":
            q.ps = [p for p in q.ps if p.tau >= cs.total_n - cs.n]

        return q

    def changepoint(self) :

        def _argmax(x) :
            return max(zip(x,range(len(x))))[1]
        if self.ql.opt > self.qr.opt:
            i = _argmax([p.get_max(self.cs) - 0.0 for p in self.ql.ps[:-1]])
            most_likely_changepoint_location = max(0, self.cs.total_n - self.cs.n + self.ql.ps[i].tau)
        else:
            i = _argmax([p.get_max(self.cs) - 0.0 for p in self.qr.ps[:-1]])
            most_likely_changepoint_location = max(0, self.cs.total_n - self.cs.n + self.qr.ps[i].tau)
        return {"stopping_time": self.cs.total_n,"changepoint": most_likely_changepoint_location}

    def update(self, y):
 
        # updating the cusums and count with the new point
        self.cs.update(y)

        # updating the value of the max of the null (for pre-change mean unkown)
        m0 = 0
        if self.qr.ps[0].theta0 is None:
            m0 = self.qr.ps[0].get_max(self.cs)

        if self.side not in  ["left", "right", "both"]:
            raise ValueError("size should be either 'both', 'right' or 'left'.")

        if self.side == "both" or self.side == "right":
            TruncatedFocus._prune(self.qr, self.cs, "right")  # true for the right pruning
            self.qr.opt = Focus._get_max_all(self.qr, self.cs, m0)
            # add a new point
            self.qr.ps.append(self.comp_func(self.cs.sn, self.cs.n, m0))
        if self.side == "both" or self.side == "left":
            TruncatedFocus._prune(self.ql, self.cs, "left")  # false for the left pruning
            self.ql.opt = Focus._get_max_all(self.ql, self.cs, m0)
            self.ql.ps.append(self.comp_func(self.cs.sn, self.cs.n, m0))


    class _TruncatedCUSUM:
        def __init__(self, sn=0, n=0, window_size=500):
            self.sn = sn
            self.n = n
            self.total_n = n
            self.rolling_sum = _RollingSum(window_size=window_size)

        def update(self, y):
            self.rolling_sum.add(y)
            self.sn = self.rolling_sum.current_sum
            self.n = self.rolling_sum.current_count
            self.total_n += 1

            if self.total_n >= 499:
                hold=1

class _RollingSum:
    def __init__(self, window_size):
        self.window_size = window_size
        self.queue = deque(maxlen=window_size)
        self.current_sum = 0  # Store the rolling sum
        self.current_count = 0

    def add(self, value):
        if len(self.queue) == self.window_size:
            self.current_sum -= self.queue[0]  # Remove oldest value from sum
        else:
            self.current_count += 1
        self.queue.append(value)
        self.current_sum += value  # Add new value to sum

    def get_sum_count(self):
        return self.current_sum, self.current_count  # Return current rolling sum



##########################
######## NPFocus #########
##########################

from collections import Counter, deque

class NPFocus:

    def __init__(self, quantiles, side = "both"):
        # Ensure that the quantiles list is not nested
        if any(isinstance(i, list) for i in quantiles):
            raise ValueError("Quantiles list should not be nested.")
        
        # init - same side for all quantiles
        if len(side) != len(quantiles):
            side = [side for _ in range(len(quantiles))]

        # initializing the bernoulli detectors        
        self.detectors = [Focus(Bernoulli(), side=s) for s in side]
        self.quantiles = quantiles

    def update(self, y):
        for (d, q) in zip(self.detectors, self.quantiles):
            d.update((y <= q) * 1)

    def statistic(self) :
        return [d.statistic() for d in self.detectors]

    def changepoint(self) :

        # Get statistics and changepoints for each detector
        stats_changepoints = [(d.statistic(), d.changepoint()) for d in self.detectors]

        # Find the detector with the highest statistic
        max_stat, max_stat_changepoint = max(stats_changepoints, key=lambda x: x[0])

        return {"stopping_time": max_stat_changepoint["stopping_time"], "changepoint": max_stat_changepoint["changepoint"], "max_stat": max_stat}


# This part is for testing purposes
if __name__ == "__main__":
  import numpy as np

  np.random.seed(0)
  Y = np.concatenate((np.random.normal(loc=0.0, scale=1.0, size=5000), np.random.normal(loc=10.0, scale=1.0, size=5000)))

  # Assuming Focus and Gaussian classes are defined elsewhere (replace with your implementation)
#   detector = TruncatedFocus(Gaussian(), window_size=500)
  detector = Focus(Gaussian())
  threshold = 10.0
  for y in Y:
      detector.update(y)
      print(detector.statistic())
      if detector.statistic() >= threshold:
          break

  result = detector.changepoint()
  print(f"We detected a changepoint at time {result['stopping_time']}.")