### How to add a new Alorithm in this framework
All the algorithms are written in **bandit.py**. 
The parent class is **Bandit()** and if you would like to test your own algorithms, please written a new class inherited the parent class.  

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

