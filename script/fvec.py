import struct

def save_fvecs_dataset(dataset: list[list[float]], path: str):
    with open(path, "wb") as f:
        for v in dataset:
            # Write 4 Byte Dimension
            dim: int = len(v)
            f.write(dim.to_bytes(4, byteorder='little'))
            # Write the vector
            for x in v:
                f.write(bytearray(struct.pack("f", x)))

def read_fvecs_dataset(path: str) -> list[list[float]]:
    with open(path, "rb") as f:
        dataset = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            v = []
            for i in range(dim):
                x = struct.unpack("f", f.read(4))[0]
                v.append(x)
            dataset.append(v)
        return dataset
    
def read_ivecs_dataset(path: str) -> list[list[float]]:
    with open(path, "rb") as f:
        dataset = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            v = []
            for i in range(dim):
                x = struct.unpack("i", f.read(4))[0]
                v.append(x)
            dataset.append(v)
        return dataset

def read_fbin_dataset(path: str) -> list[list[float]]:
    with open(path, "rb") as f:
        dataset = []
        num_point = struct.unpack("i", f.read(4))[0]
        num_dim = struct.unpack("i", f.read(4))[0]
        for _pts in range(num_point):
            v = []
            for _dim in range(num_dim):
                x = struct.unpack("f", f.read(4))[0]
                v.append(x)
            dataset.append(v)
        print(f"Finished Reading Dataset[{num_point} * {num_dim}]")
        print(f"Returning [{len(dataset)} * {len(dataset[0])}]")
        return dataset
    
if __name__ == "__main__":
    
    # An Example of Transforming fbin format to fvecs format
    src = "./data/DEEP/base.10M.fbin"
    dst =  "./data/DEEP/deep_base.fvecs"

    ds = read_fbin_dataset(src)
    save_fvecs_dataset(ds, dst)