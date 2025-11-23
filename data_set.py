import polynomial

import math
import random

def trunc_num(num: float, dec_places: int):
    return math.floor(num * 10 ** dec_places) / (10 ** dec_places)

one_hot_encoding_table = { char: index for index, char in enumerate("0123456789.,-") }

class DataSet:
    # (input, output)
    # input = one hot encoding indices
    # output = polynomial result
    training_data: list[(list[int], float)]
    validation_data: list[(list[int], float)]

    def __init__(self, input_range: float, num_training_samples: int, num_validation_samples: int):
        func = polynomial.load_polynomial("./results/polynomial.json")

        vars = { var: 0 for _, _, var in func.terms }

        all_data = []

        for _ in range(num_training_samples + num_validation_samples):
            for var in vars: vars[var] = trunc_num(input_range * random.uniform(-1, 1), 3)
            
            all_data.append((
                [one_hot_encoding_table[char] for char in ",".join(str(val) for val in vars.values())],
                func.evaluate(vars)
            ))

        self.training_data = all_data[:num_training_samples]
        self.validation_data = all_data[num_training_samples:]

def main():
    test_data_set = DataSet(10, 5, 5)

    for inp, out in test_data_set.training_data:
        inp_str = ""

        for ohe in inp:
            inp_str += str(ohe) if ohe < 10 else ".,-"[ohe - 10]

        print(f"Input String: {inp_str}")
        print(f"Output: {out}\n")

if __name__ == "__main__": main()