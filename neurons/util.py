def zero_center(data,col_name):
    col_mean = data[col_name].mean()
    data[col_name] = data[col_name] - col_mean
    return data

def normalize(data,col_name,zscore=False):
    std = data[col_name].std()
    data[col_name] = data[col_name] / std
    return data
