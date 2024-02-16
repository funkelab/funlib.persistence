from funlib.persistence.arrays.datasets import _read_voxel_size_offset, check_for_offset, check_for_voxel_size, access_parent

import pytest
from numcodecs import Zstd
import zarr
import numpy as np

@pytest.fixture(scope='session')
def filepaths(tmp_path_factory):
    path = tmp_path_factory.mktemp('test_data', numbered=False)
    test_n5_path = path / 'input/test_file.n5'
    test_zarr_path = path / 'output/test_file.zarr'

    n5_data = populate_n5file(test_n5_path, test_metadata_n5())
    zarr_data = populate_zarrfile(test_zarr_path, test_metadata_zarr())
    
    return (n5_data, zarr_data)

def test_metadata_n5():
    metadata_n5 = {"pixelResolution":{"dimensions":[5.3,4.3,3.3],
                        "unit":"nm"},
                        "ordering":"C",
                        "scales":[[1,1,1],[2,2,2]],
                        "axes":["x","y","z"],
                        "units":["nm","nm","nm"],
                        "transform" : {
                                        "axes": ["z","y","x"],
                                        "ordering": "C",
                                        "scale": [5.3, 4.3, 3.3],
                                        "translate": [4.3, 3.3, 2.3],
                                        "units": ["nm", "nm", "nm"]}
                        }
    return metadata_n5

def test_metadata_zarr():
    zarr_metadata = {
        "multiscales": [
            {
                "axes": [
                    {
                        "name": "z",
                        "type": "space",
                        "unit": "nanometer"
                    },
                    {
                        "name": "y",
                        "type": "space",
                        "unit": "nanometer"
                    },
                    {
                        "name": "x",
                        "type": "space",
                        "unit": "nanometer"
                    }
                ],
                "coordinateTransformations": [],
                "datasets": [
                    {
                        "coordinateTransformations": [
                            {
                                "scale": [3.3,4.3,5.3],
                                "type": "scale"
                            },
                            {
                                "translation": [2.3, 3.3, 4.3],
                                "type": "translation"
                            }
                        ],
                        "path": "test_data"
                    }
                ],
                "name": "",
                "version": "0.4"
            }
        ]
    }
    return zarr_metadata
    
    

#n5 test file
def populate_n5file(filepath, test_metadata_n5):
    
    store = zarr.N5Store(filepath)
    root = zarr.group(store = store, path = 'test_group', overwrite = True) 
    
    n5_data = zarr.create(store=store, 
                            path= 'test_data', 
                            shape = (100,100, 100),
                            chunks=10,
                            dtype='uint8', compressor=Zstd(level=6))
    n5_data[:] = np.random.rand(100,100, 100)
        
    n5_data.attrs.update(test_metadata_n5())
    root.attrs.update(test_metadata_n5())
  
    return n5_data
        
    
#zarr test file
def populate_zarrfile(filepaths, test_metadata_n5, test_metadata_zarr):
    zarr_path = filepaths[1]
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store = store, path = 'test_group', overwrite = True) 
    
    zarr_data = zarr.create(store=store, 
                            path= 'test_data', 
                            shape = (100,100, 100),
                            chunks=10,
                            dtype='uint8', compressor=Zstd(level=6))
    zarr_data[:] = np.random.rand(100,100, 100)
     
    zarr_data.attrs.update(test_metadata_n5()) 
    root.attrs.update(test_metadata_zarr())
   
    return zarr_data
        

def test_read_voxel_size_offset(filepaths):
    #assert voxel_size
    #assert offset 
    n5_arr = filepaths[0]
    zarr_arr = filepaths[1]
    assert _read_voxel_size_offset(n5_arr) == ([5.3, 4.3, 3.3],
                                               [4.3, 3.3, 2.3],
                                               ["nanometer","nanometer","nanometer"])
    assert _read_voxel_size_offset(zarr_arr) == ([3.3,4.3,5.3],
                                                 [2.3, 3.3, 4.3],
                                                 ["nm", "nm", "nm"])



