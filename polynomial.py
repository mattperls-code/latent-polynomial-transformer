import random
import math
import json

class Polynomial:
    # (coeff, power, var)
    terms: list[(float, int, str)]

    def __init__(self, terms: list[(float, int, str)]):
        self.terms = terms

    def evaluate(self, vars: dict[str, float]):
        return sum(coeff * vars[var] ** power for coeff, power, var in self.terms)

    def to_string(self):
        return " + ".join([f"{coeff} {var} ^ {power}" for coeff, power, var in self.terms])
    
    def to_latex(self):
        return " + ".join([f"{coeff} {var} ^ {{{power}}}" for coeff, power, var in self.terms])

# generate a multivariable polynomial of a given degree such that its range is within some desired range when evaluated in a given input range
def generate_polynomial(num_vars: int, degree: int, input_range: float, desired_range: float):
    terms = []

    num_terms = num_vars * degree

    for i in range(1, 1 + num_vars):
        var = f"x_{i}"

        for power in range(1, 1 + degree):
            # scale the contribution of each term to ensure desired range is respected
            # coeff * input_range ** power must be at most desired_range / num_terms

            max_coeff = desired_range / (num_terms * input_range ** power)
            coeff = max_coeff * math.cbrt(random.uniform(-1, 1))

            terms.append((coeff, power, var))

    return Polynomial(terms)

def load_polynomial(path: str):
    with open(path, "r") as input_file:
        return Polynomial([tuple(term) for term in json.load(input_file)])

def main():
    with open("./results/polynomial.json", "w") as output_file:
        json.dump(generate_polynomial(10, 5, 10, 1).terms, output_file, indent=4)

    new_polynomial = load_polynomial("./results/polynomial.json")

    print("New Polynomial:\n" + new_polynomial.to_string())

if __name__ == "__main__": main()