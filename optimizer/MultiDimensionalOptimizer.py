import numpy as np


class MultiDimensionalOptimizer :
    '''
    Implementation of DeepGo's nested optimization algorithm, for the multidimensional case.
    '''
    def __init__(self,w, constraints, K=0, ETA = 1.1, MAX_ITERS = 1000, ERROR_TOLERANCE = 0.01, update_eta = False ):
        '''
        Parameters
        -----------------
        w: function
            The Lipschitz continuous function to optimize.
        
        constraints: list
            List of tuples [(a_i,b_i)], representing the box constraints for each optimization variable.


        K : float, optional
            The Lipschitz constant overestimation to use during the optimization.
            If set to 0 or None, then the Lipschitz constant will be dynamically estimated using the 
            maximum norm of the gradient seen so far.

        ETA: float, optional 
            The overshoot factor to use when dynamically estimating the Lipschitz constant. ETA must be > 1.
            If `K` is not None and not 0, then a fixed Lipschitz constant overestimation will be used, and
            the Lipschitz constant will not be dynamically estimated.

        MAX_ITERS: int
            The maximum number of iterations to perform at each optimization level.
        
        ERROR_TOLERANCE: float
            Return solution when upper_bound - lower_bound < ERROR_TOLERANCE.
        
        update_eta: bool
            If set to true the overshoot factor eta will be dynamically updated during the run of the algorithm.
            eta will be exponentially decayed to `ETA` using the formula :
            self.eta = self.ETA + self.N0 * np.exp(-self.LAMBDA * self.total_iterations)

        '''
        assert ETA > 1.0 and MAX_ITERS > 0
        self.w = w; self.n = len(constraints)
        self.constraints = constraints; self.K = K
        self.originalK = self.K
        self.ETA = ETA; self.MAX_ITERS = MAX_ITERS; self.ERROR_TOLERANCE = ERROR_TOLERANCE
        self.N0 = 50 # initial value to add to self.ETA
        self.LAMBDA = 0.001  # exponential decay rate
        if self.K:
            self.update_eta = False
        else :
            self.update_eta = update_eta
        if update_eta :
            self.eta = self.ETA + self.N0
        else :
            self.eta = ETA
        self.max_gradient = 0 # the maximum gradient norm seen so far
        self.total_iterations = 0 # total number of iterations so far

    def minimize(self) :
        '''
        Minimize `self.w` subject to `self.constraints`.

        Returns
        ----------------
        minimum: float

        argmin: list
        '''
        self.K = self.originalK
        if self.K :
            return self.optimize(self.w,self.constraints, estimate_K = False) 
        return self.optimize(self.w,self.constraints, estimate_K = True)

    def maximize(self) :
        '''
        Maximize `self.w` subject to `self.constraints`.

        Returns
        --------
        maximum: float

        argmax: list
        '''
        print("MAXIMIZING")
        self.K = self.originalK
        if self.K :
            maximum, argmax = self.optimize(lambda  *xs: -self.w(*xs), self.constraints, estimate_K = False)
        else :
            maximum, argmax = self.optimize(lambda  *xs: -self.w(*xs), self.constraints, estimate_K = True)
        return -maximum, argmax
    
    def optimize(self,w,constraints, estimate_K = False) :
        '''
        Wrapper around `__optimize`.
        '''
        self.minK = np.inf
        return self.__optimize(w, constraints, [], 0 , estimate_K )

    def __optimize(self, w, constraints, arguments, depth, estimate_K = False) :
        '''
        Implementation of the nested optimization algorithm.
        
        Parameters
        ------------------------
            w: function
                The Lipschitz-continuous function to optimize.
            
            constraints: list
                List of tuples [(a_i,b_i)], representing the box constraints for each optimization variable.
            
            depth: int
                The depth of the current optimization problem.
            
            estimate_K : bool
                Set to True to dynamically estimate the Lipschitz constant, otherwise
                a fixed estimate of the Lipschitz constant will be used

        Returns
        --------------------------
        u: float
            The upper bound estimate of the true minimum.
            
        best_args: list
            The list of arguments that yielded `u`. (i.e. the argmin of the function)
        '''
        if depth  == self.n :
            return w(*arguments), arguments
        a,b = constraints[depth]
        assert a<b
        w_a, args_a = self.__optimize(w,constraints, arguments + [a], depth+1, estimate_K)
        w_b, args_b = self.__optimize(w, constraints, arguments + [b], depth+1, estimate_K)
        
        self.max_gradient = max(self.max_gradient, self.gradient(w_a,w_b,a,b) ) 
        if estimate_K :
            self.K = self.eta * self.max_gradient
        
        if self.K == 0 :
            print("WARNING: Zero gradient encountered.")
            self.K = 10.0 # set to some large constant in this case
        assert self.K!=0
        self.minK = min(self.minK, self.K)
        self.check_K()
        Y = [a,b]
        W = [w_a, w_b]
        if w_a <= w_b :
            u = w_a
            best_args = args_a
        else :
            u = w_b
            best_args = args_b
        l = self.new_z(w_a,w_b,a,b,self.K)
        Z = [l]

        iteration = 0
        i = 0 # argmin Z
        while iteration < self.MAX_ITERS  and u-l >= self.ERROR_TOLERANCE :
            assert self.K!=0
            if depth == 0 :
                print("iteration = ", iteration, " l = ",l, " u = ",u)
            y_i = self.new_y(W[i], W[i+1], Y[i], Y[i+1],self.K)
            w_i, args_i = self.__optimize(w, constraints, arguments+[y_i], depth+1, estimate_K)
            if estimate_K:
                self.max_gradient = max(self.max_gradient, self.gradient(W[i], w_i, Y[i], y_i))
                self.K = self.eta * self.max_gradient
            self.minK = min(self.minK, self.K)
            self.check_K()
            z_left = self.new_z(W[i], w_i, Y[i], y_i ,self.K)
            z_right = self.new_z(w_i,W[i+1], y_i , Y[i+1], self.K)
            Y.insert(i+1,y_i)
            W.insert(i+1, w_i)
            Z[i] = z_left
            Z.insert(i+1,z_right)
            # update current solution
            if w_i < u :
                u = w_i
                best_args = args_i
            # update for next iteration
            i = np.argmin(Z)
            l = Z[i]
            iteration+=1
            self.total_iterations+=1
            if self.total_iterations % 1000 == 0 :
                print("At total_iterations = ",self.total_iterations, " K = ", self.K, " max_gradient = ", self.max_gradient, " eta = ", self.eta)
            if estimate_K and self.update_eta:
                self.eta = self.ETA + self.N0 * np.exp(-self.LAMBDA * self.total_iterations)
        if iteration >= self.MAX_ITERS : 
                print("WARNING: Maximum number of iterations reached.")
        return u, best_args


    def check_K(self):
        '''
        Check that the best Lipschitz constant has not been underestimated.
        '''
        assert self.minK >= self.max_gradient, "ERROR: Lipschitz constant \
        was underestimated minK = %f  ,  max_gradient = %f, eta = %f" % (self.minK, self.max_gradient,self.eta)

    def new_z(self, w_a, w_b, a, b, K) :
        assert K!=0 and a!=b
        return (w_a + w_b)/ 2  - K * (b-a) / 2

    def new_y(self,w_a,w_b,a,b,K):
        assert K!=0 and a!=b
        return (a+b)/2 - (w_b-w_a) / (2 * K)

    def gradient(self,w_a,w_b,a,b):
        assert a != b
        return abs((w_b-w_a)/(b-a))


if __name__ == '__main__' :
    w = lambda x,y : 0.5 * (x**2 + 2*x*np.cos(x**2)) + (y-2)**2
    constraints = 2*[(-3,3)]
    etas = [1.2, 1.5, 2.1]
    for eta in etas:
        print("For ETA = ", eta, " : ")
        optimizer = MultiDimensionalOptimizer(w,constraints, ERROR_TOLERANCE = 0.001, ETA = eta, MAX_ITERS = 5000, K = None, update_eta = True)
        minimum , argmin = optimizer.minimize()
        print(" Minimum: ",minimum)
        print(" argmin: ", argmin)
        assert w(*argmin) == minimum
        maximum, argmax = optimizer.maximize()
        print("Maximum: ",maximum)
        print("argmax: ", argmax)
        assert w(*argmax) == maximum
        print(30*'-')