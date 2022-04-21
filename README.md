# Code to Accompany the Book "Bandit Algorithms for Website Optimization"


This repo contains code in several languages that implements several standard algorithms for solving the Multi-Armed Bandits Problem, including:

* epsilon-Greedy
* UCB1
* ETC

#  Requirement
- python 3.6  


# Getting Started

To try out this code, you can go into the Python directories and then run the demo script.

In Python, that looks like:

    python task.py

You should step through that code line-by-line to understand what the functions are doing. The book provides more in-depth explanations of how the algorithms work.

The Ruby code was contributed by Kashif Rasul. If you're interested in translating the code into another language, please submit a pull request. I will merge any new implementations as soon as I can.

# How To Adding New Algorithms: API Expectations

All the algorithms are written in **bandit.py**. 
The parent class is **Bandit()** and if you would like to test your own algorithms, please code a new class inherited the parent class.  

#### For example
```python

class Bandit:
    def __init__(self, k_arm=10, initial=0., step_size=0.1, sample_averages=False):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages 
        self.indices = np.arange(self.k) 
        self.initial = initial  

    def step(self):
        ...

class Epsilon_Greedy(Bandit):
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False):
        Bandit.__init__(k_arm, initial, step_size, sample_averages)
        self.epsilon = epsilon  

    def act(self):
        ...
```

# TODO  
- [ ] add UCB2 algorithms
- [ ] add EXP3 algorithms

# Reference
In R, there is a body of code for visualizing the results of simulations and analyzing those results. The R code would benefit from some refactoring to make it DRYer.

If you're interested in seeing how some of these algorithms would be implemented in Javascript, you should try out Mark Reid's code: http://mark.reid.name/code/bandits/

If you're looking for Java code, try Dani Sola's work: https://github.com/danisola/bandit

If you're looking for Scala code, try everpeace(Shingo Omura)'s work: https://github.com/everpeace/banditsbook-scala

If you're looking for Go code, try Rany Keddo's work: https://github.com/purzelrakete/bandit

If you're looking for Clojure code, try Paul Ingles's work: https://github.com/pingles/clj-bandit

If you're looking for Swift code, see https://github.com/crenwick/Swiper

For a Flask implementation, see https://github.com/DeaconDesperado/flask_mab