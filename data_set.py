from polynomial import Polynomial, load_polynomial

import math
import random

def trunc_num(num: float, dec_places: int):
    if num > 0: return math.floor(num * 10 ** dec_places) / (10 ** dec_places)
    else: return math.ceil(num * 10 ** dec_places) / (10 ** dec_places)

one_hot_encoding_table = { char: index for index, char in enumerate("0123456789,-") }

class DataSet:
    func: Polynomial

    # (input, output)
    # input = one hot encoding indices
    # output = polynomial result
    training_data: list[(list[int], float)]
    validation_data: list[(list[int], float)]

    def __init__(self, num_training_samples: int, num_validation_samples: int):
        self.func = load_polynomial("./results/polynomial.json")

        vars = { var: 0 for var in self.func.get_vars() }

        all_data = []

        for _ in range(num_training_samples + num_validation_samples):
            max_domain = 1 - 1e-7

            for var in vars: vars[var] = trunc_num(random.uniform(-max_domain, max_domain), 4 + random.randint(0, 4))
            
            all_data.append((
                # replace "0." so that only significant chars are used
                [one_hot_encoding_table[char] for char in ",".join(format(val, "f").replace("0.", "") for val in vars.values())],
                self.func.evaluate(vars)
            ))

        self.training_data = all_data[:num_training_samples]
        self.validation_data = all_data[num_training_samples:]

def main():
    test_data_set = DataSet(5, 5)

    for inp, out in test_data_set.validation_data:
        inp_str = ""

        for ohe in inp:
            inp_str += str(ohe) if ohe < 10 else ",-"[ohe - 10]

        print(f"Input String: {inp_str}")
        print(f"Output: {out}\n")

if __name__ == "__main__": main()