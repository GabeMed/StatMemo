import sys
import random
import itertools
DEPTH = 4
GENERATIONS = 400


def memoize(f):
    """Add a cache memory to the input function."""
    f.cache = {}

    
    def decorated_function(*args):
        if args in f.cache:
            return f.cache[args]
        else:
            f.cache[args] = f(*args)
            return f.cache[args]
    return decorated_function


def randexpr(dep, vars):
    """Create a random Boolean expression."""
    if dep == 1 or random.random() < 1.0 / (2 ** dep - 1):
        return random.choice(vars)
    if random.random() < 1.0 / 3:
        return 'not' + ' ' + randexpr(dep - 1, vars)
    else:
        return '(' + randexpr(dep - 1, vars) + ' ' + random.choice(['and', 'or']) + ' ' + randexpr(dep - 1, vars) + ')'


def randfunct(vars):
    """Create a random Boolean function. Individuals are represented _directly_ as Python functions."""
    re = randexpr(DEPTH, vars)
    temp1 = ', '
    rf = eval('lambda ' + temp1.join(vars) + ': ' + re)
    rf = memoize(rf)
    rf.geno = lambda: re
    return rf


def targetfunct(*args):
    """Parity function of any number of input variables"""
    return args.count(True) % 2 == 1


def fitness(individual, numvars):
    """Determine the fitness (error) of an individual. Lower is better."""
    fit = 0
    somelists = [[True, False] for i in range(numvars)]
    for element in itertools.product(*somelists):
        if individual(*element) != targetfunct(*element):
            fit = fit + 1
    return fit


def mutation(p, vars):
    """The mutation operator is a higher order function. The parent function is called by the offspring function."""
    temp2 = ' and '
    mintermexpr = temp2.join([random.choice([x, 'not ' + x]) for x in vars])
    temp3 = ', '
    minterm = eval('lambda ' + temp3.join(vars) + ': ' + mintermexpr)
    if random.random() < 0.5:
        offspring = lambda *x: p(*x) or minterm(*x)
        offspring = memoize(offspring)
        offspring.geno = lambda: '(' + p.geno() + ' or ' + mintermexpr + ')'
    else:
        offspring = lambda *x: p(*x) and (not minterm(*x))
        offspring = memoize(offspring)
        offspring.geno = lambda: '(' + p.geno() + ' and not ' + mintermexpr + ')'
    return offspring


def climb(numvars, vars):
    """Main function. As the landscape is always unimodal the climber can find the optimum."""
    curr = randfunct(vars)
    curr.fit = fitness(curr, numvars)
    for gen in range(GENERATIONS + 1):
        off = mutation(curr, vars)
        off.fit = fitness(off, numvars)
        if off.fit < curr.fit:
            curr = off
        if gen % 10 == 0:
            curr.fit


def main(numvars, vars):
    climb(numvars, vars)
if __name__ == '__main__':
    numvars = int(sys.argv[1])
    vars = ['x' + str(i) for i in range(numvars)]
    main(numvars, vars)

def run_tiny(numvars):                
    vars_ = tuple(f"x{i}" for i in range(numvars))   
    return main(numvars, vars_)        