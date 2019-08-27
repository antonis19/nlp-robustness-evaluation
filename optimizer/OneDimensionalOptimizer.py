import numpy as np

class  OneDimensionalOptimizer:
    '''
    Implementation of DeepGo's nested optimization algorithm for one-dimensional Lipschitz continuous functions.
    '''
    def __init__(self, w, a, b, K = 0, ETA = 1.1, MAX_ITERS = 5000, ERROR_TOLERANCE = 0.001):
            assert a<=b 
            assert ETA > 1.0 and MAX_ITERS>0
            self.w = w ; self.a = a; self.b = b; self.K = K; 
            self.ETA = ETA; self.MAX_ITERS = MAX_ITERS; self.ERROR_TOLERANCE = ERROR_TOLERANCE

    def minimize(self) :
        if self.K :
            return self.optimize(self.w,self.a,self.b,self.K) 
        return self.optimize(self.w,self.a,self.b, estimate_K = True)

    def maximize(self) :
        if self.K :
            return -self.optimize(lambda x: -self.w(x),self.a,self.b,self.K)
        return -self.optimize(lambda x: -self.w(x),self.a,self.b, estimate_K = True)

    def optimize(self, w, a, b, K = 0, estimate_K = False) :
        assert a<=b
        if estimate_K :
            K = self.ETA * self.gradient(w,a,b)
        if K == 0 :
            K = 10.0 # set to  some large constant if K is 0
        u = min(w(a),w(b))
        l = self.new_z(w,a,b,K)
        Z = [l]
        Y = [a,b]

        iteration = 0
        i = 0 # argmin Z
        while iteration < self.MAX_ITERS  and u-l >= self.ERROR_TOLERANCE :
            y_i = self.new_y(w,Y[i],Y[i+1],K)
            if estimate_K:
                K = max(K, self.ETA * self.gradient(w,Y[i],y_i))
            z_left = self.new_z(w,Y[i],y_i,K)
            z_right = self.new_z(w,y_i,Y[i+1],K)
            Y.insert(i+1,y_i)
            Z[i] = z_left
            Z.insert(i+1,z_right)
            assert sorted(Y) == Y
            # update for new iteration
            i = np.argmin(Z)
            l = Z[i] 
            u = min(u, w(y_i))
            iteration+=1
        print("TERMINATING IN ITERATION = ",iteration)
        if iteration == self.MAX_ITERS : 
            print("WARNING: Maximum number of iterations reached.")
        return u

    def new_z(self, w, a, b, K) :
        assert K > 0 
        return (w(b) + w(a))/ 2  - K * (b-a) / 2

    def new_y(self, w, a, b, K) :
        assert K > 0
        return (a+b)/2 - (w(b)-w(a)) / (2 * K)

    def gradient(self,w,a,b):
        assert a!=b
        return abs((w(b)-w(a))/(b-a))

if __name__ == "__main__" :
    w = lambda x : 0.5 * (x**2 + 2*x*np.cos(x**2)); a = -3; b = 3
    etas = [1.01, 1.1, 1.4, 2.2]
    for eta in etas:
        print('For ETA = ', eta, ' : ')
        optimizer = OneDimensionalOptimizer(w,a,b, ETA = eta)
        minimum = optimizer.minimize()
        print(" Minimum: ", minimum)
        maximum = optimizer.maximize()
        print(" Maximum: ", maximum)
        print(30*'-')
    # True minimum is -0.3826
    # True maximum is 7.2333