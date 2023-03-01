import scripts.utils as utils

# setup the model and other dependencies
setup = utils.Setup()

# data to test on
test_data = [
    [1, 1, 30, 1, 130_000, 2, 2],   # 5
    [1, 1, 34, 0, 96_000, 0, 1],   # 1
    [1, 0, 24, 1, 76_000, 0, 1],   # 3
    [0, 1, 55, 1, 106_000, 1, 1],   # 4
    [0, 0, 43, 2, 114_000, 2, 2],   # 0
]

# make predictions
result = setup.predict(test_data)
# print result
print(result)
