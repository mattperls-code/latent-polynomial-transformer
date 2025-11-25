import random
import math
import json

class Polynomial:
    # (coeff, (power, var))
    terms: list[(float, list[(int, str)])]

    def __init__(self, terms: list[(float, list[(int, str)])]):
        self.terms = terms

    def get_vars(self):
        return set([var for _, monomials in self.terms for _, var in monomials])

    def evaluate(self, vars: dict[str, float]):
        return sum(coeff * sum([vars[var] ** power for power, var in monomials]) for coeff, monomials in self.terms)

    def to_string(self):
        return " + ".join([f"{coeff} " + " ".join([f"{var} ^ {power}" for power, var in monomials]) for coeff, monomials in self.terms])
    
    def to_latex(self):
        return " + ".join([f"{coeff} " + " ".join([f"{var} ^ {{{power}}}" for power, var in monomials]) for coeff, monomials in self.terms])

# generate a random multivariable polynomial such that D: (-1, 1) and R: (-1, 1)
def generate_polynomial(num_vars: int, degree: int):
    max_domain = 1 - 1e-10
    max_range = 1 - 1e-10

    # num unique ways to pick degree for each var
    num_terms = num_vars ** degree

    desired_range_per_term = max_domain / num_terms

    terms = []

    for i in range(0, num_terms):
        monomials = []

        monomials_product = 1

        for j in range(0, num_vars):
            power = (i % degree) + 1
            var = f"x_{j + 1}"
            
            i -= i % degree
            i /= degree

            monomials.append((power, var))

            monomials_product *= max_range ** power

        # ensure contribution of this term is within desired_range_per_term
        max_coeff = desired_range_per_term / monomials_product

        # skew distribution toward -max or +max
        coeff = max_coeff * math.cbrt(math.cbrt(random.uniform(-1, 1)))

        terms.append((coeff, monomials))

    return Polynomial(terms)

def load_polynomial(path: str):
    with open(path, "r") as input_file:
        return Polynomial([tuple(term) for term in json.load(input_file)])

def main():
    with open("./results/polynomial.json", "w") as output_file:
        json.dump(generate_polynomial(4, 4).terms, output_file, indent=4)

    new_polynomial = load_polynomial("./results/polynomial.json")

    print("New Polynomial:\n" + new_polynomial.to_string())

if __name__ == "__main__": main()